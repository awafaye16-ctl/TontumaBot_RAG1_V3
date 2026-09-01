import re

# Mots wolof EXCLUSIFS — n'existent PAS en français courant
# On retire volontairement : ci, la, li, lo, wi, bi, gi, si, di, bu, su, na, am,
# dem, carte, mairie, dakar (communs au français ou trop ambigus)
WOLOF_WORDS = {
    # pronoms / personnes
    "man", "yow", "moom", "nun", "yeen", "ñoom",
    # marqueurs verbaux wolof (inexistants en français)
    "dama", "danga", "dafa", "dañu", "danuy", "dinaa", "dinanga",
    "muy", "maa", "nga",
    # verbes wolof distinctifs
    "beug", "bëgg", "jëm", "nekk", "soxor",
    "wax", "xam", "toog", "dox", "jeend", "seet",
    "jox", "fay", "bind", "jàng", "daldi", "waaw", "déedéet",
    # mots de question wolof
    "ana", "naka", "ndax", "loolu", "lëndëm", "kuy", "fooy",
    # mots courants wolof EXCLUSIFS (pas en français)
    "fii", "bés", "bam", "ñu",
    "wout", "kayitu", "juddu", "jërejëf", "dimbal", "soxla",
    "metit", "tontu", "xiif", "naari",
    "jigéen", "góor", "xale", "bëi",
    "kër", "dëkk", "mbedd", "daara", "naan",
    "njëkk", "ginnaaw", "kanam", "xorom", "ndox",
    "nit", "nitt", "jàmm", "xam-xam", "yëgël",
    "suma", "ñu",
    # mots administratifs wolof
    "dokimaan",
    # connecteurs / particules wolof exclusifs
    "ngir", "wante", "waaye", "nde",
    "ak", "doon", "lekki", "woon",
    # autres formes wolof distinctives
    "bii", "bëi", "mag", "góor",
}

# Mots français courants à exclure explicitement (évite faux positifs)
FRENCH_STOPWORDS = {
    "carte", "mairie", "comment", "quels", "pour", "obtenir",
    "une", "les", "des", "que", "qui", "quoi", "est", "sont",
    "dans", "avec", "sur", "par", "au", "aux", "de", "du",
    "je", "tu", "il", "nous", "vous", "ils", "me", "ma", "mon",
    "ce", "se", "sa", "le", "la", "les",
}


def detect_language(text: str) -> str:
    words = set(re.findall(r"[\wàâäéèêëîïôöùûüç]+", text.lower()))
    if not words:
        return "fr"

    # Mots wolof exclusifs présents (hors stopwords français)
    wolof_hits = len(words & WOLOF_WORDS - FRENCH_STOPWORDS)

    # Seuil : au moins 1 mot wolof exclusif
    # ET la proportion de mots wolof / total mots doit être > 10%
    # (évite de détecter "carte d'identité ci" comme wolof)
    if wolof_hits >= 1:
        ratio = wolof_hits / len(words)
        if ratio >= 0.10 or wolof_hits >= 2:
            return "wo"

    return "fr"
