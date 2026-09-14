"""Router d'intention : procedure vs orientation.

Classification par similarité d'embeddings : la question est comparée à des
phrases-prototypes de chaque intention avec l'embedder déjà chargé par le
vectorstore (`paraphrase-multilingual-MiniLM-L12-v2`). Aucun modèle
supplémentaire n'est chargé, et le modèle étant multilingue la classification
fonctionne aussi bien sur une question restée en wolof.

Convention du projet :
  orientation — lieu, direction, service à contacter, coût/tarif
                (cf. les tags `lieu` / `service` / `cout` de FilteredSearch)
  procedure   — démarches, documents, étapes à suivre
                (seule intention qui déclenche la génération d'un QR code)

Le routeur par mots-clés reste disponible (`detect_intent_keywords`) et sert de
repli si l'embedder est indisponible.
"""
import os
import re
import sys

from journal import journal

_log = journal("intent")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Phrases-prototypes ───────────────────────────────────────────────────────
# Elles décrivent l'intention par l'exemple : en ajouter une suffit à couvrir
# une nouvelle formulation, sans toucher à la logique.
PROTOTYPES: dict[str, list[str]] = {
    "orientation": [
        "Où se trouve le service de cardiologie ?",
        "À quel étage est la radiologie ?",
        "Dans quel bâtiment se trouve la maternité ?",
        "Comment aller à la pharmacie depuis l'entrée principale ?",
        "Comment se rendre au laboratoire ?",
        "Je cherche le guichet des admissions.",
        "Quel est le numéro de téléphone de l'accueil ?",
        "Quel service dois-je contacter pour ce problème ?",
        "Combien coûte une consultation ?",
        "Quel est le tarif d'une radiographie ?",
        # Formulations wolof : le bot est wolof-first et l'embedder est
        # multilingue, ce qui garde le routeur juste si la traduction WO→FR
        # échoue ou est contournée.
        "Fan la nekk sarwis bi ?",
        "Ñaata la konsultasioŋ bi ?",
    ],
    "procedure": [
        "Comment obtenir un certificat médical ?",
        "Quels documents faut-il fournir pour une hospitalisation ?",
        "Quelles sont les étapes pour se faire rembourser ?",
        "Comment prendre rendez-vous avec un spécialiste ?",
        "Quelle est la procédure d'admission aux urgences ?",
        "Comment payer ma facture d'hospitalisation ?",
        "Que dois-je faire pour récupérer mes résultats d'analyse ?",
        "Comment déposer un dossier de prise en charge ?",
        "Naka laa war a def ngir am kayit bi ?",
    ],
}

DEFAULT_INTENT = "procedure"

# Cache des embeddings de prototypes : [(intention, vecteur), ...]
_proto: list[tuple[str, list[float]]] | None = None


# ── Repli par mots-clés ──────────────────────────────────────────────────────
ORIENTATION_KEYWORDS = {
    "où", "aller", "déposer", "rendre", "adresse", "bureau", "guichet",
    "service", "administration", "contacter", "adresser", "localiser",
    "quel service", "quelle administration", "quel ministère", "commissariat",
    "mairie", "lieu", "endroit", "situé", "située", "combien", "coûte",
    "coûtent", "gratuit", "coût", "tarif", "prix", "frais", "bâtiment",
    "batiment", "étage", "etage", "immeuble", "centre", "agence", "poste",
    "ambassade",
}
_WORD_ONLY = {
    "où", "aller", "rendre", "mairie", "lieu", "prix", "coût", "frais",
    "combien", "étage", "etage",
}


def _contains_keyword(q: str, kw: str) -> bool:
    if kw in _WORD_ONLY:
        return re.search(rf"\b{re.escape(kw)}\b", q) is not None
    return kw in q


def detect_intent_keywords(query_fr: str) -> str:
    """Routeur historique par mots-clés — repli si l'embedder est indisponible."""
    q = query_fr.lower()
    if any(_contains_keyword(q, kw) for kw in ORIENTATION_KEYWORDS):
        return "orientation"
    return DEFAULT_INTENT


# ── Routeur par embeddings ───────────────────────────────────────────────────
def load_prototypes() -> list[tuple[str, list[float]]]:
    """Embeddings des prototypes, calculés une seule fois (appelable au warmup)."""
    global _proto
    if _proto is None:
        import vectorstore  # import lazy : évite un cycle à l'import du module
        labels  = [lab for lab, phrases in PROTOTYPES.items() for _ in phrases]
        phrases = [p for phrases in PROTOTYPES.values() for p in phrases]
        vectors = vectorstore.get_embedder().encode(
            phrases, normalize_embeddings=True
        ).tolist()
        _proto = list(zip(labels, vectors))
    return _proto


def _best_scores(embedding: list[float]) -> dict[str, float]:
    """Similarité cosinus max entre la question et les prototypes de chaque intention.

    Les vecteurs étant normalisés, le cosinus se réduit à un produit scalaire.
    """
    scores = {label: -1.0 for label in PROTOTYPES}
    for label, vec in load_prototypes():
        sim = sum(a * b for a, b in zip(embedding, vec))
        if sim > scores[label]:
            scores[label] = sim
    return scores


def detect_intent(query_fr: str, query_embedding: list[float] | None = None) -> str:
    """Retourne 'orientation' ou 'procedure'.

    Args:
        query_fr:        question (français, ou wolof — le modèle est multilingue).
        query_embedding: embedding normalisé déjà calculé pour cette question.
                         Fourni par le pipeline, il rend la classification gratuite.
    """
    try:
        if query_embedding is None:
            import vectorstore
            query_embedding = vectorstore.embed_query(query_fr)
        scores = _best_scores(query_embedding)
        return max(scores, key=scores.get)
    except Exception as e:  # embedder indisponible → repli mots-clés
        _log.error(f"Embedder indisponible ({e}) — repli mots-clés")
        return detect_intent_keywords(query_fr)
