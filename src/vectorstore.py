"""Vectorstore V3 — ChromaDB + embeddings + BM25 hybride + MMR.

Pipeline de recherche :
  1. Hybrid search   BM25 (lexical) + vectoriel (sémantique)  → top-K candidats
  2. MMR             Maximum Marginal Relevance               → diversité, anti-redondance
  3. Reranker        cross-encoder                            → tri final par pertinence

Toutes les fonctions de recherche retournent des tuples
  (chunk_id, texte, score, metadata)
pour que le pipeline puisse tracer chaque étape.
"""
import hashlib
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import settings  # noqa: E402

import chromadb
import numpy as np

EMBED_MODEL = settings.EMBED_MODEL
COLLECTION  = "tontuma_v3"
DB_DIR      = os.path.join(settings.BASE_DIR, "data", "chroma")

_embedder   = None
_client     = None
_collection = None


# =============================================================================
#  Embedding
# =============================================================================

def get_embedder() -> "SentenceTransformer":
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer  # import lazy
        print(f"[VS] Chargement embedder ({EMBED_MODEL})...")
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def _embed(texts: list[str]) -> list[list[float]]:
    return get_embedder().encode(texts, normalize_embeddings=True).tolist()


def _embed_single(text: str) -> list[float]:
    return _embed([text])[0]


# =============================================================================
#  Collection ChromaDB
# =============================================================================

def get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        os.makedirs(DB_DIR, exist_ok=True)
        _client = chromadb.PersistentClient(path=DB_DIR)
    return _client


def get_collection():
    global _collection
    if _collection is not None:
        return _collection
    client = get_client()
    hnsw_meta = {"hnsw:space": "cosine"}
    try:
        _collection = client.get_collection(COLLECTION)
        # Si le modèle d'embedding a changé → on purge pour rester cohérent
        if (_collection.metadata or {}).get("embed_model") != EMBED_MODEL:
            client.delete_collection(COLLECTION)
            _collection = client.create_collection(
                COLLECTION,
                metadata={**hnsw_meta, "embed_model": EMBED_MODEL},
            )
    except Exception:
        _collection = client.create_collection(
            COLLECTION,
            metadata={**hnsw_meta, "embed_model": EMBED_MODEL},
        )
    return _collection


# =============================================================================
#  Écriture
# =============================================================================

def add_documents(chunks: list[str], metadatas: list[dict]) -> int:
    """Ajoute (ou met à jour) des chunks dans ChromaDB.

    L'ID est un hash MD5 de (texte + metadata) pour garantir l'idempotence :
    ingérer deux fois le même document ne crée pas de doublons.
    """
    if not chunks:
        return 0
    coll = get_collection()
    ids  = [
        "doc-" + hashlib.md5((c + "|" + str(m)).encode()).hexdigest()[:16]
        for c, m in zip(chunks, metadatas)
    ]
    embeddings = _embed(chunks)
    coll.upsert(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
    return len(chunks)


# =============================================================================
#  Recherche vectorielle pure
# =============================================================================

def vector_search(
    query: str,
    k: int = 10,
    filter_meta: Optional[dict] = None,
) -> list[tuple[str, str, float, dict]]:
    """Recherche vectorielle cosinus dans ChromaDB.

    Returns:
        Liste de (chunk_id, texte, score_similarité, metadata), triée par score desc.
    """
    coll = get_collection()
    if coll.count() == 0:
        return []

    kwargs = {"query_embeddings": [_embed_single(query)], "n_results": k,
              "include": ["documents", "metadatas", "distances"]}
    if filter_meta:
        kwargs["where"] = filter_meta

    res = coll.query(**kwargs)
    out = []
    for doc, meta, dist, cid in zip(
        res["documents"][0], res["metadatas"][0],
        res["distances"][0],  res["ids"][0],
    ):
        out.append((cid, doc, 1.0 - dist, meta))  # distance cosinus → similarité
    return out


# =============================================================================
#  Recherche hybride BM25 + vectorielle
# =============================================================================

def hybrid_search(
    query: str,
    k: int   = 10,
    alpha: float = 0.4,
) -> list[tuple[str, str, float, dict]]:
    """Fusionne BM25 (lexical) et vectoriel (sémantique).

    score_final = alpha * score_BM25 + (1 - alpha) * score_vectoriel

    alpha = 0.4 → légère préférence au sémantique (meilleur sur les questions
    administratives paraphrasées).

    Returns:
        Liste de (chunk_id, texte, score_fusionné, metadata), top-K.
    """
    coll = get_collection()
    if coll.count() == 0:
        return []

    from retrieval.hybrid import HybridRetriever

    all_data  = coll.get(include=["documents", "metadatas"])
    ids       = all_data["ids"]
    texts     = all_data["documents"]
    metas     = all_data["metadatas"]

    # BM25 sur tous les chunks
    retriever  = HybridRetriever(texts)
    bm25_scores = dict(retriever._bm25_search(query, k=k * 3))

    # Vectoriel
    id_to_pos  = {cid: i for i, cid in enumerate(ids)}
    vec_scores = {}
    for cid, _, s, _ in vector_search(query, k=k * 3):
        if cid in id_to_pos:
            vec_scores[id_to_pos[cid]] = s

    # Fusion
    results = []
    for idx in set(bm25_scores) | set(vec_scores):
        s_b = bm25_scores.get(idx, 0.0)
        s_v = vec_scores.get(idx, 0.0)
        results.append((ids[idx], texts[idx], alpha * s_b + (1 - alpha) * s_v, metas[idx]))

    results.sort(key=lambda x: x[2], reverse=True)
    return results[:k]


# =============================================================================
#  MMR — Maximum Marginal Relevance
# =============================================================================

def mmr_search(
    query: str,
    k: int      = 5,
    fetch_k: int = 20,
    lambda_mmr: float = 0.6,
) -> list[tuple[str, str, float, dict]]:
    """Récupère fetch_k candidats par hybride puis sélectionne k chunks
    par MMR pour maximiser pertinence et diversité.

    MMR(d) = lambda * sim(query, d) - (1 - lambda) * max_sim(d, already_selected)

    lambda_mmr = 0.6 : légèrement biaisé vers la pertinence,
                       mais avec bonne diversité (évite les chunks redondants).

    Returns:
        Liste de (chunk_id, texte, score_mmr, metadata), k éléments distincts.
    """
    candidates = hybrid_search(query, k=fetch_k)
    if not candidates:
        return []

    # ── Déduplication par texte exact avant MMR ───────────────────────────
    seen_texts: set[str] = set()
    deduped: list = []
    for item in candidates:
        txt = item[1].strip()
        if txt not in seen_texts:
            seen_texts.add(txt)
            deduped.append(item)
    candidates = deduped

    embedder    = get_embedder()
    q_vec       = np.array(embedder.encode([query], normalize_embeddings=True)[0])
    cand_texts  = [c[1] for c in candidates]
    cand_vecs   = embedder.encode(cand_texts, normalize_embeddings=True)

    selected_idx  : list[int]   = []
    selected_vecs : list[np.ndarray] = []
    remaining_idx : list[int]   = list(range(len(candidates)))

    for _ in range(min(k, len(candidates))):
        best_idx   = -1
        best_score = -float("inf")

        for i in remaining_idx:
            relevance = float(np.dot(q_vec, cand_vecs[i]))
            if selected_vecs:
                redundancy = max(
                    float(np.dot(cand_vecs[i], sv)) for sv in selected_vecs
                )
            else:
                redundancy = 0.0

            mmr_score = lambda_mmr * relevance - (1 - lambda_mmr) * redundancy
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx   = i

        selected_idx.append(best_idx)
        selected_vecs.append(cand_vecs[best_idx])
        remaining_idx.remove(best_idx)

    return [
        (candidates[i][0], candidates[i][1], float(np.dot(q_vec, cand_vecs[i])), candidates[i][3])
        for i in selected_idx
    ]


# =============================================================================
#  Recherche filtrée par métadonnée (pour l'intention "orientation")
# =============================================================================

def filtered_search(
    query: str,
    filter_meta: dict,
    k: int = 5,
) -> list[tuple[str, str, float, dict]]:
    """Restreint la recherche aux chunks ayant des métadonnées spécifiques.

    Exemple : filtered_search(query, {"document_id": "abc123"}, k=5)
    """
    return vector_search(query, k=k, filter_meta=filter_meta)


# =============================================================================
#  Gestion des documents
# =============================================================================

def all_documents() -> list[dict]:
    """Retourne la liste des documents indexés (dédoublonnés par document_id)."""
    coll = get_collection()
    data = coll.get(include=["documents", "metadatas"])
    docs: dict = {}
    for doc, meta in zip(data["documents"], data["metadatas"]):
        did = meta.get("document_id", "?")
        if did not in docs:
            docs[did] = {
                "id":     did,
                "title":  meta.get("title", did),
                "source": meta.get("source", ""),
                "chunks": 0,
                "added":  meta.get("added", ""),
            }
        docs[did]["chunks"] += 1
    return list(docs.values())


def delete_document(document_id: str) -> int:
    coll = get_collection()
    data = coll.get(where={"document_id": document_id})
    ids  = data["ids"]
    if ids:
        coll.delete(ids=ids)
    return len(ids)


def clear_all() -> int:
    coll = get_collection()
    n    = coll.count()
    if n:
        coll.delete(ids=coll.get()["ids"])
    return n


def count() -> int:
    return get_collection().count()
