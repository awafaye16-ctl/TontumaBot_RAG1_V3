"""Normalisation du texte wolof avant synthèse vocale.

Oolel-Voices ne lit correctement que du wolof écrit en toutes lettres. Sondé
par aller-retour (synthèse puis re-transcription HuBERT-CTC), tout le reste se
dégrade — et parfois détruit l'énoncé entier, pas seulement le mot fautif :

    « Ubbeeku na ci 8:00 ba 17:00. »  ->  « ay tëur ba sept cenbrle ba sept... »
    « Demal ca APIX. »                ->  « xala bou bayi ahi demal kë »
    « Ñu jox na 50% ci njëg li. »     ->  « ñu jox la seki s quox ci njëg li »
    « Këyit bi (kopi bi) ... »        ->  « u bi kopp bi ... »   (« Këyit » perdu)

Témoin, en wolof pur : « Demal ca bérébu liggéeyukaay bi. » revient intact.
Le moteur est donc sain ; c'est l'écriture qu'il faut lui préparer.

`pour_synthese()` est le point d'entrée unique, appelé juste avant le TTS. Le
texte affiché à l'écran n'est pas touché : il garde ses chiffres et ses sigles,
plus lisibles à l'œil.

Les tables de ce module relèvent de l'usage wolof et ont été arbitrées avec
l'équipe : heures en 24 h suffixées de `waxtu`, pourcentages en `pursaan`,
sigles rendus par une graphie explicite. Elles sont isolées en tête pour rester
corrigibles sans toucher à la logique.
"""
import re

from .nombres import _LIAISON_MOT, _wo_nombre, en_lettres_wolof

from journal import journal

_log = journal("prononciation")

# ── Tables arbitrées ──────────────────────────────────────────────────────

# Graphie prononçable des sigles. Un sigle absent de cette table est laissé
# tel quel ET signalé (voir `sigles_inconnus`) : mieux vaut un défaut visible
# qu'une prononciation inventée.
#
# Graphies retenues après comparaison par aller-retour : pour APIX, « apiks »
# revient plus fidèlement que « aapiks », et une graphie espacée (« aa piks »)
# est à proscrire — elle fusionne avec le mot qui précède.
SIGLES = {
    "APIX":  "apiks",
    "CFA":   "seefaa",
    "FCFA":  "seefaa",
    "ID":    "idantite",
    "CNI":   "se en i",
    "PDF":   "pe de ef",
    "QR":    "ku er",
}

_MOT_HEURE   = "waxtu"      # 8:00  -> juróom-ñetti waxtu
_MOT_PERCENT = "pursaan"    # 50%   -> juróomi fukki pursaan

# Nom des minutes : non arbitré. Tant qu'il est vide, une heure à minutes non
# nulles est rendue « H waxtu ak M », sans nommer l'unité.
_MOT_MINUTE  = ""


def _wo_liaison(n: int) -> str:
    """Nombre wolof sous sa forme de liaison, celle qui précède un nom compté.

    « juróom-ñett » devient « juróom-ñetti waxtu », « juróomi fukk » devient
    « juróomi fukki pursaan ».
    """
    mots = _wo_nombre(n).split()
    if mots and mots[-1] in _LIAISON_MOT:
        mots[-1] = _LIAISON_MOT[mots[-1]]
    return " ".join(mots)


# ── Heures ────────────────────────────────────────────────────────────────
#  Traitées AVANT les nombres : sans cela « 8:00 » se lit « juróom-ñett:tus »,
#  le deux-points restant collé entre deux mots.

# L'espace éventuelle avant les minutes appartient au groupe optionnel :
# sinon « 8h ba 17h » consomme l'espace suivante et rend « waxtuba ».
_HEURE = re.compile(r"\b(\d{1,2})\s*(?::|h|H)(?:\s*(\d{2}))?\b")


def _heures(texte: str) -> str:
    def _rendre(m: re.Match) -> str:
        heure = int(m.group(1))
        if heure > 23:
            return m.group()
        minute = int(m.group(2)) if m.group(2) else 0
        if minute > 59:
            return m.group()
        dit = f"{_wo_liaison(heure)} {_MOT_HEURE}"
        if minute:
            suffixe = f" {_MOT_MINUTE}".rstrip()
            dit += f" ak {_wo_liaison(minute) if _MOT_MINUTE else _wo_nombre(minute)}{suffixe}"
        return dit
    return _HEURE.sub(_rendre, texte)


# ── Pourcentages ──────────────────────────────────────────────────────────

_POURCENT = re.compile(r"\b(\d{1,3})\s*%")


def _pourcentages(texte: str) -> str:
    return _POURCENT.sub(
        lambda m: f"{_wo_liaison(int(m.group(1)))} {_MOT_PERCENT}", texte
    )


# ── Sigles ────────────────────────────────────────────────────────────────

_SIGLE = re.compile(r"\b[A-ZÀ-Þ]{2,}\b")


def _sigles(texte: str) -> tuple[str, list[str]]:
    """Remplace les sigles connus ; retourne aussi la liste des inconnus."""
    inconnus: list[str] = []

    def _rendre(m: re.Match) -> str:
        sigle = m.group()
        if sigle in SIGLES:
            return SIGLES[sigle]
        inconnus.append(sigle)
        return sigle

    return _SIGLE.sub(_rendre, texte), inconnus


# ── Ponctuation ───────────────────────────────────────────────────────────
#  Les parenthèses ne brouillent pas leur contenu mais le mot qui les précède
#  (« Këyit bi (kopi bi) » a perdu « Këyit »). On les remplace par des virgules :
#  la pause reste, la gêne disparaît.

_REMPLACEMENTS = [
    # NLLB bascule les horaires en 12 heures et laisse un « a.m. » / « p.m. »
    # anglais derrière. Une fois l'heure réalignée sur le français (24 h), ce
    # suffixe est faux autant qu'imprononçable : on le retire.
    (re.compile(r"\s*\b[ap]\s*\.\s*m\s*\.", re.IGNORECASE), ""),
    (re.compile(r"\s*\(\s*"), ", "),
    (re.compile(r"\s*\)\s*"), ", "),
    (re.compile(r"\s*[\[\]{}]\s*"), ", "),
    (re.compile(r"\s*/\s*"), ", "),
    (re.compile(r"\s*[«»\"']\s*"), " "),
    (re.compile(r"\s*[–—]\s*"), ", "),
    (re.compile(r"[*#_`|<>]"), ""),
    (re.compile(r",\s*(?=[.!?;:])"), ""),   # virgule juste avant une fin
    (re.compile(r"(,\s*){2,}"), ", "),      # virgules accumulées
    (re.compile(r"\s{2,}"), " "),
]


def _ponctuation(texte: str) -> str:
    for motif, remplacement in _REMPLACEMENTS:
        texte = motif.sub(remplacement, texte)
    return texte.strip()


# ── Point d'entrée ────────────────────────────────────────────────────────

def pour_synthese(texte: str) -> tuple[str, dict]:
    """Prépare un texte wolof pour Oolel-Voices.

    Retourne le texte prononçable et un bilan destiné à la trace :
        sigles_inconnus  sigles sans graphie connue, laissés tels quels
        modifie          True si le texte diffère de l'entrée
    """
    if not texte:
        return texte, {"sigles_inconnus": [], "modifie": False}

    sortie = _heures(texte)
    sortie = _pourcentages(sortie)
    # Les nombres AVANT les sigles : la conversion monétaire absorbe l'unité
    # (« 1000 francs CFA » -> « ñaari téeméer dërëm »). Traiter les sigles
    # d'abord transformerait CFA en « seefaa », que l'unité ne reconnaîtrait
    # plus — il resterait un « seefaa » orphelin après le montant.
    sortie = en_lettres_wolof(sortie)
    sortie, inconnus = _sigles(sortie)
    sortie = _ponctuation(sortie)

    return sortie, {
        "sigles_inconnus": sorted(set(inconnus)),
        "modifie":         sortie != texte,
    }


if __name__ == "__main__":
    CAS = [
        "Ubbeeku na ci 8:00 ba 17:00.",
        "Ubbeeku na ci 8h ba 17h30.",
        "Demal ca APIX.",
        "Wutal sa ID bu baax.",
        "Ñu jox na 50% ci njëg li.",
        "Këyit bi (kopi bi) danga koy joxe.",
        "Njëg li 1000 francs CFA la.",
        "Demal ci sarwis bi lundi/vendredi.",
        "Boyet limat 4 ubbeeku na ci 8:00 ba 17:00.",
        "Demal ca bérébu liggéeyukaay bi.",
        "Demal ca ONFP.",
    ]
    for source in CAS:
        sortie, bilan = pour_synthese(source)
        marque = f"   [inconnus: {bilan['sigles_inconnus']}]" if bilan["sigles_inconnus"] else ""
        print(f"  {source}\n    -> {sortie}{marque}")