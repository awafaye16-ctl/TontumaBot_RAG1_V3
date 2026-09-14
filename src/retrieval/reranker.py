"""Reranker : cross-encoder pour reranking des résultats du retriever."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import settings

from journal import journal

_log = journal("reranker")

_model = None


def load_model():
    global _model
    if _model is not None:
        return _model
    from sentence_transformers import CrossEncoder  # import lazy
    import device as _dev
    dev = _dev.resolve("reranker")
    _log.info(f"Chargement ({settings.RERANKER_MODEL}) sur {dev.upper()}...")
    _model = CrossEncoder(settings.RERANKER_MODEL, device=dev)
    _log.info("Chargé.")
    return _model


def rerank(query: str, documents: list[str], top_k: int = None) -> list[tuple[int, float, str]]:
    """Rerank `documents` par rapport à `query`.
    Retourne [(index_original, score, texte)] triés par pertinence décroissante."""
    if not documents:
        return []
    if top_k is None:
        top_k = settings.RERANKER_TOP_K
    model = load_model()
    pairs = [(query, doc) for doc in documents]
    scores = model.predict(pairs)
    indexed = list(enumerate(scores))
    indexed.sort(key=lambda x: x[1], reverse=True)
    return [(idx, float(score), documents[idx]) for idx, score in indexed[:top_k]]
