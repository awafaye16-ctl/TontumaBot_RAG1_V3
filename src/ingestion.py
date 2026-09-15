"""Ingestion V3.1 — Pipeline RAG rigoureux, borné à une organisation.

Étapes :
  1. Extraction texte  (PDF, TXT, MD)
  2. Chunking sémantique  (sur frontières de phrases, taille + overlap configurables)
  3. Métadonnées enrichies par fragment  (organization_id, document_id, titre,
     source, catégorie, index, nb_mots, date)
  4. Embedding + stockage dans la base de l'organisation

Le chunking respecte les frontières de phrases (.!?) et les titres Markdown (##)
pour éviter de couper un contexte sémantique au milieu d'une idée.

Identifiant du document
-----------------------
`document_id` est fourni par l'appelant (le backend) et conservé tel quel. C'est
ce qui permet de republier une démarche administrative dont le contenu change
sans empiler les versions : le même identifiant désigne le même document pour
toute sa durée de vie.

Le remplacement est une SUPPRESSION suivie d'une insertion, jamais une fusion.
Il ne peut pas reposer sur l'idempotence de `add_documents` : l'identifiant d'un
fragment dérive de son texte, donc un contenu modifié produit de nouveaux
fragments qui s'ajouteraient à côté des anciens au lieu de les remplacer.
"""
import hashlib
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import vectorstore  # noqa: E402

from journal import journal

_log = journal("ingestion")

class ExtractionImpossible(ValueError):
    """Le fichier n'a pas pu être lu : PDF corrompu, tronqué, ou chiffré.

    Distinct du PDF scanné, qui se lit très bien mais ne contient aucun texte —
    celui-là produit simplement 0 fragment. Les deux finissent en 400 côté API,
    mais avec des messages différents : l'agent qui a déposé le fichier doit
    pouvoir distinguer « votre document est une image » de « votre fichier est
    abîmé ».
    """


# ── Paramètres de chunking ────────────────────────────────────────────────
CHUNK_SIZE    = 512   # caractères max par chunk
CHUNK_OVERLAP = 80    # chevauchement pour conserver le contexte entre chunks
MIN_CHUNK_LEN = 40    # on ignore les micro-fragments vides


# =============================================================================
#  1. Extraction de texte
# =============================================================================

def extract_text(path: str) -> str:
    """Extrait le texte brut d'un fichier (TXT, MD, PDF)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _extract_pdf(path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _extract_pdf(path: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise RuntimeError("pypdf requis pour les PDF : pip install pypdf")

    # pypdf lève une famille d'erreurs qui lui est propre (PdfReadError,
    # DependencyError pour un fichier chiffré…), et qui ne dérive d'aucune
    # exception standard. Sans ce filet, un PDF abîmé — cas banal d'un dépôt
    # interrompu — ressortait en 500 avec une trace pypdf, là où le contrat
    # promet un 400 explicite.
    try:
        reader = PdfReader(path)
        pages  = [page.extract_text() or "" for page in reader.pages]
    except Exception as e:  # noqa: BLE001
        raise ExtractionImpossible(
            f"PDF illisible ({type(e).__name__}). Le fichier est peut-être "
            f"corrompu, tronqué ou protégé par mot de passe."
        ) from e
    return "\n\n".join(pages)


# =============================================================================
#  2. Chunking sémantique
# =============================================================================

def _normalize(text: str) -> str:
    """Normalise les espaces et les lignes vides multiples."""
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _split_into_sentences(text: str) -> list[str]:
    """Découpe en phrases sur .!? et titres Markdown."""
    # Titres Markdown → frontière de phrase
    text = re.sub(r"(#{1,6}\s+[^\n]+)", r"\1.", text)
    # Séparation sur .!? suivis d'un espace ou newline
    parts = re.split(r"(?<=[.!?…])\s+|\n\n+", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int    = CHUNK_OVERLAP,
) -> list[str]:
    """Découpe le texte en chunks cohérents.

    Algorithme :
      - accumule les phrases jusqu'à atteindre chunk_size
      - quand le seuil est atteint, sauvegarde le chunk
      - repart avec les `overlap` derniers caractères du chunk précédent
        (pour conserver le contexte entre chunks adjacents)
    """
    text = _normalize(text)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    sentences = _split_into_sentences(text)
    chunks: list[str] = []
    current           = ""

    for sentence in sentences:
        candidate = (current + " " + sentence).strip() if current else sentence
        if len(candidate) > chunk_size and current:
            # Sauvegarde le chunk courant
            chunks.append(current.strip())
            # Repart avec le chevauchement
            tail    = current[-overlap:] if len(current) > overlap else current
            current = (tail + " " + sentence).strip()
        else:
            current = candidate

    if current.strip() and len(current.strip()) >= MIN_CHUNK_LEN:
        chunks.append(current.strip())

    # Filtre les micro-fragments
    return [c for c in chunks if len(c) >= MIN_CHUNK_LEN]


# =============================================================================
#  3. Métadonnées enrichies
# =============================================================================

def _make_metadata(
    organization_id: str,
    document_id: str,
    title: str,
    source: str,
    category: str | None,
    chunk_index: int,
    total_chunks: int,
    chunk_text: str,
    added: str,
) -> dict:
    """Construit les métadonnées d'un fragment.

    Champs :
      organization_id — organisation propriétaire. Redondant avec le répertoire
                        de la base, et c'est volontaire : ça permet d'auditer
                        après coup à qui appartient un fragment, et de migrer
                        vers une autre stratégie d'isolation sans réindexer.
      document_id   — identifiant du document parent, fourni par l'appelant
      title         — titre du document
      source        — nom du fichier, clé de l'objet, ou 'manual'
      category      — catégorie transmise par l'appelant (facultative)
      chunk_index   — position du fragment dans le document (0-based)
      total_chunks  — nombre total de fragments du document
      nb_chars      — longueur du fragment en caractères
      nb_words      — nombre de mots (approximatif)
      added         — timestamp ISO 8601 UTC de l'ingestion

    ChromaDB refuse les valeurs nulles en métadonnée : une catégorie absente est
    stockée en chaîne vide et renormalisée en `None` à la lecture.
    """
    return {
        "organization_id": organization_id,
        "document_id":  document_id,
        "title":        title,
        "source":       source,
        "category":     category or "",
        "chunk_index":  chunk_index,
        "total_chunks": total_chunks,
        "nb_chars":     len(chunk_text),
        "nb_words":     len(chunk_text.split()),
        "added":        added,
    }


# =============================================================================
#  4. Points d'entrée publics
# =============================================================================

def _document_id_par_defaut(title: str, text: str) -> str:
    """Identifiant de repli quand l'appelant n'en fournit pas.

    Dérivé du titre et du début du texte, comme en V3.0 : deux ingestions
    identiques retombent sur le même identifiant. Ce repli ne sert qu'aux appels
    manuels et aux tests — le backend fournit toujours le sien, sans quoi une
    republication dont le texte a changé créerait un second document au lieu de
    remplacer le premier.
    """
    return hashlib.md5((title + text[:300]).encode()).hexdigest()[:16]


def ingest_text(
    organization_id: str,
    text: str,
    title: str = "Document manuel",
    source: str = "manual",
    document_id: str | None = None,
    category: str | None = None,
) -> dict:
    """Ingère du texte brut dans la base de l'organisation.

    Si `document_id` désigne un document déjà indexé, celui-ci est ENTIÈREMENT
    remplacé : ses fragments sont supprimés avant l'insertion des nouveaux.

    Returns:
        dict {document_id, chunks, replaced, title, category, added}
        `chunks` à 0 signifie qu'aucun texte exploitable n'a été trouvé ; dans
        ce cas rien n'est supprimé — un document déjà indexé n'est jamais perdu
        au profit d'une version vide.
    """
    org         = vectorstore.valider_organisation(organization_id)
    document_id = (document_id or "").strip() or _document_id_par_defaut(title, text)

    chunks = chunk_text(text)
    if not chunks:
        return {"document_id": document_id, "chunks": 0, "replaced": False,
                "title": title, "category": category, "added": None}

    # Remplacement : on supprime AVANT d'insérer. L'opération n'est pas atomique
    # — il existe une fenêtre de quelques secondes pendant laquelle le document
    # est absent de l'index. C'est assumé et documenté dans le contrat backend.
    supprimes = vectorstore.delete_document(org, document_id)
    if supprimes:
        _log.info("organisation %s : document %s remplacé (%d fragments retirés)",
                  org, document_id, supprimes)

    now   = datetime.now(timezone.utc).isoformat()
    total = len(chunks)

    metadatas = [
        _make_metadata(org, document_id, title, source, category, i, total, c, now)
        for i, c in enumerate(chunks)
    ]
    n = vectorstore.add_documents(org, chunks, metadatas)
    _log.info("organisation %s : '%s' → %d fragments (document_id=%s)",
              org, title, n, document_id)
    return {"document_id": document_id, "chunks": n, "replaced": bool(supprimes),
            "title": title, "category": category, "added": now}


def ingest_file(
    organization_id: str,
    path: str,
    title: str | None = None,
    document_id: str | None = None,
    category: str | None = None,
    source: str | None = None,
) -> dict:
    """Ingère un fichier (TXT, MD, PDF) dans la base de l'organisation.

    `source` permet de conserver la provenance réelle — la clé de l'objet MinIO —
    plutôt que le nom du fichier temporaire dans lequel il a été téléchargé.

    Returns:
        Même dict que `ingest_text`. `chunks` à 0 sur un PDF sans texte
        extractible (document scanné sans OCR), qui est le cas le plus fréquent
        en production.
    """
    text = extract_text(path)
    nom  = os.path.basename(path)
    if not text.strip():
        return {"document_id": (document_id or "").strip() or None, "chunks": 0,
                "replaced": False, "title": title or nom, "category": category,
                "added": None}
    return ingest_text(
        organization_id,
        text,
        title       = title or nom,
        source      = source or nom,
        document_id = document_id,
        category    = category,
    )


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) < 3:
        print("usage : python src/ingestion.py <organization_id> <fichier> [...]")
        raise SystemExit(2)
    org = _sys.argv[1]
    for p in _sys.argv[2:]:
        r = ingest_file(org, p)
        print(f"{p} → {r['chunks']} fragments indexés (document_id={r['document_id']})")
