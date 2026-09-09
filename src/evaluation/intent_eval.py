"""Évaluation du router d'intention.

Compare les approches sur un jeu de questions annotées :
  keywords   — routeur historique par mots-clés
  embeddings — similarité aux prototypes via l'embedder du vectorstore (actuel)
  llm        — classification par le LLM Groq (optionnel : --llm)

Usage :
    python -m evaluation.intent_eval           # keywords vs embeddings
    python -m evaluation.intent_eval --llm     # ajoute le LLM Groq
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from intent.router import detect_intent, detect_intent_keywords  # noqa: E402

# ── Jeu de test ──────────────────────────────────────────────────────────────
# Convention : orientation = lieu / service à contacter / coût,
#              procedure   = démarches, documents, étapes.
CAS_FR: list[tuple[str, str]] = [
    # ── orientation : localisation ──
    ("Où se trouve le service de cardiologie ?",             "orientation"),
    ("À quel étage est la radiologie ?",                     "orientation"),
    ("Dans quel bâtiment est la maternité ?",                "orientation"),
    ("Je cherche la pharmacie",                              "orientation"),
    ("Comment aller au laboratoire d'analyses ?",            "orientation"),
    ("Le guichet des admissions est de quel côté ?",         "orientation"),
    ("Puis-je avoir le numéro de l'accueil ?",               "orientation"),
    ("Quel service dois-je contacter pour une réclamation ?", "orientation"),
    # ── orientation : coût (convention du projet) ──
    ("Combien coûte une consultation en dermatologie ?",     "orientation"),
    ("Quel est le tarif d'un scanner ?",                     "orientation"),
    # ── procedure : démarches ──
    ("Comment obtenir un certificat médical ?",              "procedure"),
    ("Quels documents pour une hospitalisation ?",           "procedure"),
    ("Quelle est la procédure de remboursement ?",           "procedure"),
    ("Comment prendre rendez-vous avec un cardiologue ?",    "procedure"),
    ("Quelles sont les étapes pour se faire opérer ?",       "procedure"),
    ("Comment payer ma facture ?",                           "procedure"),
    ("Que faire pour récupérer mes résultats d'analyse ?",   "procedure"),
    ("Comment inscrire mon enfant à la vaccination ?",       "procedure"),
    ("Quelles pièces fournir pour une prise en charge ?",    "procedure"),
    ("Comment faire une demande de dossier médical ?",       "procedure"),
]

# Questions restées en wolof : vérifient que le modèle multilingue tient
# même sans traduction préalable.
CAS_WO: list[tuple[str, str]] = [
    ("Fan la nekk kabinet doktoor bi ?",        "orientation"),
    ("Fan la ëpp cardiologie ?",                "orientation"),
    ("Naka laa war a am kayitu doktoor ?",      "procedure"),
    ("Ñaata la consultation bi ?",              "orientation"),
]


def _evaluer(nom: str, fn, cas: list[tuple[str, str]], verbeux: bool = True) -> float:
    justes, t0 = 0, time.perf_counter()
    erreurs = []
    for question, attendu in cas:
        obtenu = fn(question)
        if obtenu == attendu:
            justes += 1
        else:
            erreurs.append((question, attendu, obtenu))
    duree = (time.perf_counter() - t0) * 1000
    score = justes / len(cas) * 100
    print(f"\n{nom:12s} {justes}/{len(cas)}  ({score:.0f} %)  "
          f"— {duree:.0f} ms au total, {duree / len(cas):.1f} ms/question")
    if verbeux and erreurs:
        for question, attendu, obtenu in erreurs:
            print(f"   !! attendu={attendu:12s} obtenu={obtenu:12s} | {question}")
    return score


def _llm_intent(question: str) -> str:
    """Classification par le LLM Groq (option C) — un appel réseau par question."""
    from generation.llm import generate
    prompt = (
        "Classe la question d'un usager d'hôpital dans une seule catégorie.\n"
        "- orientation : trouver un lieu, un service à contacter, ou connaître un tarif\n"
        "- procedure : démarches à accomplir, documents à fournir, étapes à suivre\n"
        f"Question : {question}\n"
        "Réponds par un seul mot : orientation ou procedure."
    )
    reponse = str(generate(prompt, [])).strip().lower()
    return "orientation" if "orientation" in reponse else "procedure"


def main() -> None:
    avec_llm = "--llm" in sys.argv
    cas = CAS_FR + CAS_WO

    print("=" * 70)
    print(f"Évaluation du router d'intention — {len(CAS_FR)} questions FR "
          f"+ {len(CAS_WO)} WO")
    print("=" * 70)

    resultats = {
        "keywords":   _evaluer("keywords",   detect_intent_keywords, cas),
        "embeddings": _evaluer("embeddings", detect_intent,          cas),
    }
    if avec_llm:
        resultats["llm"] = _evaluer("llm", _llm_intent, cas)

    print("\n" + "=" * 70)
    for nom, score in sorted(resultats.items(), key=lambda x: -x[1]):
        print(f"  {nom:12s} {score:.0f} %")
    print("=" * 70)


if __name__ == "__main__":
    main()
