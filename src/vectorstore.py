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
_corpus     = None   # cache : corpus complet + index BM25 (invalidé à l'écriture)


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


def embed_query(text: str) -> list[float]:
    """Embedding normalisé de la requête (à calculer une seule fois par question)."""
    return _embed_single(text)


# =============================================================================
#  Cache du corpus (index BM25 + embeddings) — reconstruit uniquement à l'écriture
# =============================================================================

def _get_corpus() -> dict:
    """Charge une seule fois le corpus complet depuis ChromaDB et construit
    l'index BM25. Réutilisé pour toutes les requêtes jusqu'à la prochaine
    écriture (add/delete/clear), qui invalide le cache via `_invalidate_corpus`.

    Évite de recharger tous les documents et de reconstruire l'index BM25 à
    chaque question (auparavant fait 2× par requête).
    """
    global _corpus
    if _corpus is not None:
        return _corpus

    coll = get_collection()
    data = coll.get(include=["documents", "metadatas", "embeddings"])
    ids   = data["ids"]
    texts = data["documents"]
    metas = data["metadatas"]

    embeddings = data.get("embeddings")
    embeddings = np.asarray(embeddings) if embeddings is not None and len(embeddings) else None

    bm25 = None
    if texts:
        from retrieval.hybrid import HybridRetriever
        bm25 = HybridRetriever(texts)

    _corpus = {
        "ids":        ids,
        "texts":      texts,
        "metas":      metas,
        "embeddings": embeddings,
        "bm25":       bm25,
        "id_to_pos":  {cid: i for i, cid in enumerate(ids)},
    }
    return _corpus


def _invalidate_corpus() -> None:
    global _corpus
    _corpus = None


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
    _invalidate_corpus()
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
    query_embedding: Optional[list[float]] = None,
) -> list[tuple[str, str, float, dict]]:
    """Fusionne BM25 (lexical) et vectoriel (sémantique).

    score_final = alpha * score_BM25 + (1 - alpha) * score_vectoriel

    alpha = 0.4 → légère préférence au sémantique (meilleur sur les questions
    administratives paraphrasées).

    L'index BM25 et les embeddings du corpus sont mis en cache (`_get_corpus`) :
    ils ne sont pas reconstruits à chaque requête. L'embedding de la requête peut
    être fourni via `query_embedding` pour éviter de le recalculer.

    Returns:
        Liste de (chunk_id, texte, score_fusionné, metadata), top-K.
    """
    corpus = _get_corpus()
    ids, texts, metas = corpus["ids"], corpus["texts"], corpus["metas"]
    if not ids:
        return []

    # ── BM25 (index caché) ────────────────────────────────────────────────
    bm25_scores = dict(corpus["bm25"]._bm25_search(query, k=k * 3))

    # ── Vectoriel (embeddings cachés, similarité cosinus = produit scalaire
    #    car les vecteurs sont normalisés) ──────────────────────────────────
    vec_scores = {}
    embeddings = corpus["embeddings"]
    if embeddings is not None:
        if query_embedding is None:
            query_embedding = _embed_single(query)
        sims  = embeddings @ np.asarray(query_embedding)
        top_n = min(k * 3, len(sims))
        for idx in np.argpartition(sims, -top_n)[-top_n:]:
            vec_scores[int(idx)] = float(sims[idx])
    else:
        # Corpus sans embeddings en cache → repli sur la requête ChromaDB
        id_to_pos = corpus["id_to_pos"]
        for cid, _, s, _ in vector_search(query, k=k * 3):
            if cid in id_to_pos:
                vec_scores[id_to_pos[cid]] = s

    # ── Fusion ────────────────────────────────────────────────────────────
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

def mmr_from_candidates(
    candidates: list[tuple[str, str, float, dict]],
    query_embedding: list[float],
    k: int = 5,
    lambda_mmr: float = 0.6,
) -> list[tuple[str, str, float, dict]]:
    """Applique le MMR sur des candidats DÉJÀ récupérés (pas de nouvelle recherche).

    MMR(d) = lambda * sim(query, d) - (1 - lambda) * max_sim(d, already_selected)

    Les vecteurs des candidats sont récupérés depuis le cache d'embeddings du
    corpus (`_get_corpus`) ; on ne les ré-encode donc pas.

    Returns:
        Liste de (chunk_id, texte, score_mmr, metadata), k éléments distincts.
    """
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

    q_vec     = np.asarray(query_embedding)
    corpus    = _get_corpus()
    id_to_pos = corpus["id_to_pos"]
    emb       = corpus["embeddings"]

    # Vecteurs des candidats : depuis le cache si possible, sinon ré-encodage
    if emb is not None and all(c[0] in id_to_pos for c in candidates):
        cand_vecs = np.stack([emb[id_to_pos[c[0]]] for c in candidates])
    else:
        cand_vecs = get_embedder().encode(
            [c[1] for c in candidates], normalize_embeddings=True
        )

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


def mmr_search(
    query: str,
    k: int      = 5,
    fetch_k: int = 20,
    lambda_mmr: float = 0.6,
    query_embedding: Optional[list[float]] = None,
) -> list[tuple[str, str, float, dict]]:
    """Récupère fetch_k candidats par hybride puis sélectionne k chunks par MMR.

    Wrapper de compatibilité : effectue UN seul `hybrid_search` puis délègue à
    `mmr_from_candidates`. L'embedding de la requête est calculé une seule fois.
    """
    if query_embedding is None:
        query_embedding = _embed_single(query)
    candidates = hybrid_search(query, k=fetch_k, query_embedding=query_embedding)
    return mmr_from_candidates(candidates, query_embedding, k=k, lambda_mmr=lambda_mmr)


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
        _invalidate_corpus()
    return len(ids)


def clear_all() -> int:
    coll = get_collection()
    n    = coll.count()
    if n:
        coll.delete(ids=coll.get()["ids"])
        _invalidate_corpus()
    return n


def count() -> int:
    return get_collection().count()
