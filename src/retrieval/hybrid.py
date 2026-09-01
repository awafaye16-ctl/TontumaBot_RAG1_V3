"""Recherche hybride BM25 + vectoriel."""
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    return text.lower().split()


class HybridRetriever:
    def __init__(self, documents: list[str], embedder=None, weights=(0.5, 0.5)):
        self.documents = documents
        self.embedder = embedder
        self._tokenized = [tokenize(d) for d in documents]
        self.bm25 = BM25Okapi(self._tokenized)
        self.weights = weights
        self._embeddings = None

    def _bm25_search(self, query: str, k: int) -> list[tuple[int, float]]:
        q_tokens = set(tokenize(query))
        scores = self.bm25.get_scores(tokenize(query))
        matches = [
            (idx, float(s))
            for idx, s in enumerate(scores)
            if q_tokens & set(self._tokenized[idx])
        ]
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:k]

    def _vector_search(self, query: str, k: int) -> list[tuple[int, float]]:
        if self.embedder is None:
            return []
        if self._embeddings is None:
            self._embeddings = self.embedder(self.documents)
        import numpy as np
        q_vec = self.embedder([query])[0]
        scores = np.dot(self._embeddings, q_vec)
        ranked = sorted(enumerate(scores.tolist()), key=lambda x: x[1], reverse=True)
        return [(idx, float(s)) for idx, s in ranked[:k]]

    def search(self, query: str, k: int = 5, alpha: float = None) -> list[tuple[int, float]]:
        if alpha is None:
            alpha = self.weights[0]
        bm25_ranked = dict(self._bm25_search(query, k=k))
        vec_ranked = dict(self._vector_search(query, k=k))
        results = []
        for idx in set(bm25_ranked) | set(vec_ranked):
            s_bm25 = bm25_ranked.get(idx, 0.0)
            s_vec = vec_ranked.get(idx, 0.0)
            results.append((idx, alpha * s_bm25 + (1 - alpha) * s_vec))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
