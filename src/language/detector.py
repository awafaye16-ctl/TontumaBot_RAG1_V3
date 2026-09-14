"""Détection de la langue d'entrée : wolof ou français.

La borne reçoit la langue par bouton pour la voix, mais pas au clavier : c'est
ici que se joue le sort d'une question tapée.

L'ancienne version cherchait des mots wolof « exclusifs » dans une liste écrite
à la main. Mesuré sur 20 questions réelles, elle en classait 7 en français —
« sama Bopp bi dafay metti » n'y déclenchait AUCUNE correspondance : la table
contenait `suma` mais pas `sama`, `dafa` mais pas `dafay`, `metit` mais pas
`metti`. Une liste de formes exactes ne résiste pas aux variantes graphiques ni
aux suffixes.

On marque donc des points sur trois signaux indépendants, en comparant à ceux
du français :

  lexique     mots de contenu propres au wolof                      (fort)
  grammaire   particules et marqueurs — bi, ci, nga, bu, yi...      (moyen)
  orthographe ñ, ë, ó, et voyelles doublées (aa, ee, óo)            (moyen)

Aucun signal n'est décisif seul : « carte d'identité ci » contient une
particule wolof mais reste du français, et le score français l'emporte.
"""
import re

# ── Lexique wolof — mots de contenu ───────────────────────────────────────
#  Les variantes graphiques courantes figurent côte à côte (sama/suma,
#  metti/metit) : l'orthographe du wolof n'est pas fixée dans l'usage.
WOLOF_WORDS = {
    # pronoms et personnes
    "man", "yow", "moom", "nun", "yeen", "ñoom", "sama", "suma", "seen",
    # marqueurs verbaux
    "dama", "damay", "danga", "dangay", "dafa", "dafay", "dañu", "danuy",
    "dinaa", "dinanga", "dinga", "muy", "maa", "naa", "ngay",
    # verbes courants
    "beug", "bëgg", "bëggee", "bëggo", "jëm", "jëme", "nekk", "soxor",
    "wax", "xam", "toog", "dox", "jeend", "seet", "gis", "gise", "indi",
    "jox", "fay", "bind", "jàng", "daldi", "waaw", "déedéet", "dem", "dugal",
    "jot", "jotte", "mën", "mëna", "def", "defe", "am", "amee", "sàkku",
    "topp", "jaar", "wut", "wutal", "woo", "wootee", "jokk", "jokkool",
    "jëfandikoo", "yóbbu", "teewal", "waneel", "wane", "yegsi", "faj", "faju",
    # interrogatifs
    "ana", "naka", "ndax", "loolu", "lëndëm", "kuy", "fooy", "lan", "ban",
    "fan", "ñaata", "kan",
    # noms courants
    "fii", "bés", "bam", "wout", "kayitu", "kayit", "këyit", "juddu",
    "jërejëf", "dimbal", "soxla", "metit", "metti", "tontu", "xiif", "naari",
    "jigéen", "góor", "xale", "bopp", "yaram", "wërgi", "feebar", "fajkat",
    "fajukaay", "ospitaal", "lopitaal", "opitaal", "doktoor", "malaad",
    "kër", "dëkk", "mbedd", "daara", "naan", "xaalis", "njëg", "waxtu",
    "njëkk", "ginnaaw", "kanam", "xorom", "ndox", "bérébu", "béréb",
    "nit", "nitt", "jàmm", "yëgël", "liggéey", "liggéeyukaay", "sarwis",
    "dokimaan", "nimero", "buro", "boyet", "etaas", "taax", "batimaa",
    # connecteurs
    "ngir", "wante", "waaye", "nde", "walla", "wala", "ak", "doon", "woon",
    "ginaaw", "balaa", "laata", "bala",
    # formes distinctives
    "bii", "bëi", "mag", "néew", "jamp", "saasi", "altine", "aljuma",
}

# ── Grammaire wolof — particules et marqueurs ─────────────────────────────
#  Trop ambigus isolément pour trancher, mais leur accumulation est un signal
#  net : une phrase wolof en contient toujours plusieurs.
WOLOF_GRAMMAIRE = {
    "bi", "gi", "ji", "mi", "si", "wi", "li", "yi", "ci", "ba", "bu", "su",
    "mu", "ñu", "nga", "na", "laa", "la", "lay", "ma", "mooy", "moo", "yu",
    "yoy", "ay", "aw", "ab", "war", "wara", "di", "doy", "sa", "soo", "boo",
    "koy", "ko", "leen", "len", "nañu", "naa", "ne", "ni", "yépp", "lépp",
}

# ── Français — mots grammaticaux fréquents ────────────────────────────────
FRENCH_WORDS = {
    "le", "la", "les", "de", "des", "du", "un", "une", "et", "est", "sont",
    "pour", "avec", "dans", "sur", "par", "au", "aux", "que", "qui", "quoi",
    "comment", "quel", "quels", "quelle", "quelles", "où", "quand", "combien",
    "je", "tu", "il", "elle", "nous", "vous", "ils", "elles", "me", "ma",
    "mon", "mes", "ce", "cette", "ces", "se", "sa", "ses", "son", "leur",
    "faut", "faire", "obtenir", "aller", "prendre", "voudrais", "puis",
    "dois", "peut", "peux", "carte", "mairie", "document", "documents",
    "service", "rendez", "vous", "identité", "naissance", "extrait",
    "pas", "plus", "très", "bien", "aussi", "mais", "donc", "sans",
}

# ── Orthographe ───────────────────────────────────────────────────────────
#  ñ, ë et ó n'appartiennent pratiquement pas au français (hors emprunts).
_LETTRES_WOLOF = re.compile(r"[ñëóŋ]")
#  Le wolof double abondamment ses voyelles ; le français très rarement.
_VOYELLES_DOUBLES = re.compile(r"(aa|ee|ii|oo|uu|ëe|óo)")
#  Accents propres au français.
_ACCENTS_FR = re.compile(r"[éèêçûôîâœ]")

_MOT = re.compile(r"[\w'àâäéèêëîïôöùûüçñóŋ]+")


def scores(text: str) -> dict:
    """Détail du calcul — utile pour diagnostiquer un classement douteux."""
    bas  = (text or "").lower()
    mots = set(_MOT.findall(bas))
    if not mots:
        return {"wolof": 0, "francais": 0, "lexique": 0, "grammaire": 0,
                "orthographe": 0, "mots_fr": 0, "n_mots": 0}

    lexique     = len(mots & WOLOF_WORDS)
    grammaire   = len(mots & WOLOF_GRAMMAIRE - FRENCH_WORDS)
    orthographe = min(len(_LETTRES_WOLOF.findall(bas)), 4) \
                  + min(len(_VOYELLES_DOUBLES.findall(bas)), 4)
    mots_fr     = len(mots & FRENCH_WORDS) + min(len(_ACCENTS_FR.findall(bas)), 3)

    return {
        "wolof":       lexique * 3 + grammaire * 2 + orthographe,
        "francais":    mots_fr * 2,
        "lexique":     lexique,
        "grammaire":   grammaire,
        "orthographe": orthographe,
        "mots_fr":     mots_fr,
        "n_mots":      len(mots),
    }


def detect_language(text: str) -> str:
    """'wo' ou 'fr'. Le français reste le repli quand rien ne tranche."""
    s = scores(text)
    return "wo" if s["wolof"] > s["francais"] else "fr"


# ── Contrôle d'une langue déclarée ────────────────────────────────────────
#  Sur le chemin vocal, la langue n'est pas détectée mais annoncée : bouton de
#  la borne, sélecteur du chat. Elle choisit le moteur STT avant même qu'il y
#  ait un texte, et rien en aval ne la remet en cause. Un usager qui se trompe
#  de bouton obtient donc une transcription par le mauvais moteur, puis une
#  réponse dans la mauvaise langue, sans que rien ne le signale.
#
#  Ce contrôle ne corrige pas — il ne peut pas : le mal est fait au moment de
#  la transcription. Il rend l'erreur visible.
#
#  Le seuil existe parce que `detect_language` retombe sur 'fr' quand rien ne
#  tranche : sans lui, « Waaw. » déclarée en wolof lèverait une alerte à chaque
#  fois. On exige donc un signal net — le gagnant doit marquer au moins
#  _SEUIL_ALERTE points ET doubler le perdant. À titre de repère, un seul mot
#  wolof reconnu vaut 3 points et un mot français 2.
_SEUIL_ALERTE = 6


def contredit(declaree: str, text: str) -> dict | None:
    """La transcription contredit-elle la langue annoncée ?

    Retourne None si tout concorde ou si le signal est trop faible pour
    conclure ; sinon un bilan {declaree, detectee, scores} destiné à la trace
    et à l'affichage.
    """
    declaree = (declaree or "").lower()[:2]
    if declaree not in ("wo", "fr"):
        return None

    s = scores(text)
    gagnant, perdant = ("wo", "fr") if s["wolof"] > s["francais"] else ("fr", "wo")
    haut = max(s["wolof"], s["francais"])
    bas  = min(s["wolof"], s["francais"])

    if gagnant == declaree:
        return None
    if haut < _SEUIL_ALERTE or haut < bas * 2:
        return None

    return {
        "declaree": declaree,
        "detectee": gagnant,
        "scores":   {"wolof": s["wolof"], "francais": s["francais"],
                     "n_mots": s["n_mots"]},
    }
