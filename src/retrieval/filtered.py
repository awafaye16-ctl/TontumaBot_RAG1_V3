"""Recherche filtrée par métadonnées (orientation)."""


class FilteredSearch:
    def __init__(self, documents: list[dict]):
        self.documents = documents

    def search(self, query_fr: str, k: int = 3) -> list[dict]:
        q = query_fr.lower()
        tag = None
        if any(w in q for w in ["où", "ou", "lieu", "déposer", "adresse", "aller", "guichet", "bureau"]):
            tag = "lieu"
        elif any(w in q for w in ["quel service", "administration", "contacter", "ministère", "mairie"]):
            tag = "service"
        elif any(w in q for w in ["coût", "prix", "tarif", "combien", "gratuit"]):
            tag = "cout"
        if tag:
            matches = [d for d in self.documents if d.get("meta", {}).get(tag)]
            if matches:
                return matches[:k]
        scored = []
        for d in self.documents:
            s = sum(1 for w in q.split() if w in d["text"].lower())
            if s:
                scored.append((s, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:k]]
