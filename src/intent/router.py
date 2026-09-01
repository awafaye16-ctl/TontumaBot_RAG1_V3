"""Router d'intention : procedure vs orientation."""
import re

ORIENTATION_KEYWORDS = {
    "où", "aller", "déposer", "rendre", "adresse", "bureau", "guichet",
    "service", "administration", "contacter", "adresser", "localiser",
    "quel service", "quelle administration", "quel ministère", "commissariat",
    "mairie", "lieu", "endroit", "situé", "située", "combien", "coûte",
    "coûtent", "gratuit", "coût", "tarif", "prix", "frais", "batiment",
    "immeuble", "centre", "agence", "poste", "ambassade",
}
_WORD_ONLY = {"où", "aller", "rendre", "mairie", "lieu", "prix", "coût", "frais", "combien"}


def _contains_keyword(q: str, kw: str) -> bool:
    if kw in _WORD_ONLY:
        return re.search(rf"\b{re.escape(kw)}\b", q) is not None
    return kw in q


def detect_intent(query_fr: str) -> str:
    q = query_fr.lower()
    if any(_contains_keyword(q, kw) for kw in ORIENTATION_KEYWORDS):
        return "orientation"
    return "procedure"
