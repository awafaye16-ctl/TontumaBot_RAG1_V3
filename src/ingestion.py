"""Ingestion V3 — Pipeline RAG rigoureux.

Étapes :
  1. Extraction texte  (PDF, TXT, MD)
  2. Chunking sémantique  (sur frontières de phrases, taille + overlap configurables)
  3. Métadonnées enrichies par chunk  (document_id, titre, source, index, nb_tokens, date)
  4. Embedding + stockage ChromaDB  (via vectorstore.add_documents)

Le chunking respecte les frontières de phrases (.!?) et les titres Markdown (##)
pour éviter de couper un contexte sémantique au milieu d'une idée.
"""
import hashlib
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import vectorstore  # noqa: E402

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
    reader = PdfReader(path)
    pages  = [page.extract_text() or "" for page in reader.pages]
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
    document_id: str,
    title: str,
    source: str,
    chunk_index: int,
    total_chunks: int,
    chunk_text: str,
    added: str,
) -> dict:
    """Construit les métadonnées d'un chunk.

    Champs :
      document_id   — identifiant unique du document parent
      title         — titre du document
      source        — nom du fichier ou 'manual'
      chunk_index   — position du chunk dans le document (0-based)
      total_chunks  — nombre total de chunks du document
      nb_chars      — longueur du chunk en caractères
      nb_words      — nombre de mots (approximatif)
      added         — timestamp ISO 8601 UTC de l'ingestion
    """
    return {
        "document_id":  document_id,
        "title":        title,
        "source":       source,
        "chunk_index":  chunk_index,
        "total_chunks": total_chunks,
        "nb_chars":     len(chunk_text),
        "nb_words":     len(chunk_text.split()),
        "added":        added,
    }


# =============================================================================
#  4. Points d'entrée publics
# =============================================================================

def ingest_text(text: str, title: str = "Document manuel", source: str = "manual") -> int:
    """Ingère du texte brut dans ChromaDB.

    Returns:
        Nombre de chunks indexés.
    """
    chunks = chunk_text(text)
    if not chunks:
        return 0

    now         = datetime.now(timezone.utc).isoformat()
    document_id = hashlib.md5((title + text[:300]).encode()).hexdigest()[:16]
    total       = len(chunks)

    metadatas = [
        _make_metadata(document_id, title, source, i, total, c, now)
        for i, c in enumerate(chunks)
    ]
    return vectorstore.add_documents(chunks, metadatas)


def ingest_file(path: str, title: str = None) -> int:
    """Ingère un fichier (TXT, MD, PDF) dans ChromaDB.

    Returns:
        Nombre de chunks indexés.
    """
    text = extract_text(path)
    if not text.strip():
        return 0
    return ingest_text(
        text,
        title  = title or os.path.basename(path),
        source = os.path.basename(path),
    )


if __name__ == "__main__":
    import sys as _sys
    for p in _sys.argv[1:]:
        n = ingest_file(p)
        print(f"{p} → {n} chunks indexés")
