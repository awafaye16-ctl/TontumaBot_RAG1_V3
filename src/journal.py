"""Journalisation — une trace de terminal sur laquelle on peut déboguer.

Le projet écrivait ses traces avec `print()` : pas d'horodatage, pas de niveau,
pas de moyen de baisser le volume, et surtout aucun lien entre une ligne et la
requête qui l'a produite. Sur une borne, deux usagers peuvent parler en même
temps — FastAPI exécute le pipeline dans un fil d'exécution séparé — et les
lignes des deux requêtes s'entrelacent sans qu'on puisse les démêler.

Une ligne ressemble à ceci :

    10:42:31.128  INFO   a3f1  tts        morceau 1/3 (72 car.) → 4.7 s en 23.5 s
    └ heure à la ms      └ niveau  └ requête  └ composant

L'horloge va à la milliseconde parce que tout ce qu'on débogue ici est une
affaire de latence. L'identifiant de requête tient en quatre caractères : assez
pour distinguer, assez court pour ne pas manger la ligne.

Réglages (.env) :
    LOG_LEVEL=INFO      DEBUG | INFO | WARNING | ERROR
    LOG_FILE=           chemin d'un fichier ; vide = terminal seulement
    LOG_COULEUR=auto    auto | oui | non

Usage :
    from journal import journal
    log = journal("tts")
    log.info("morceau %d/%d en %.1f s", i, total, duree)

    from journal import contexte_requete
    with contexte_requete("a3f1"):
        ...        # toutes les lignes émises ici portent a3f1
"""
import contextvars
import logging
import os
import sys
import uuid
import warnings
from pathlib import Path

# ── Identifiant de requête ────────────────────────────────────────────────
#  Une variable de contexte plutôt qu'une globale : elle est propre à chaque
#  fil d'exécution ET à chaque tâche asynchrone, ce qu'une globale ne sait pas
#  faire. Sans elle, deux requêtes simultanées mélangeraient leurs traces.
_requete: contextvars.ContextVar[str] = contextvars.ContextVar("requete", default="----")

# Derniers segments de nom qui ne désignent pas un composant mais une catégorie.
_SEGMENTS_GENERIQUES = {"error", "access", "main", "core", "utils"}

_NIVEAUX = {"DEBUG": logging.DEBUG, "INFO": logging.INFO,
            "WARNING": logging.WARNING, "ERROR": logging.ERROR}

# Bibliothèques bavardes. Elles parlent d'elles-mêmes, pas du service : leurs
# avertissements de dépréciation noyaient les traces utiles au point qu'il
# fallait les filtrer au grep pour lire quoi que ce soit.
_BAVARDES = {
    "urllib3": logging.WARNING,
    "httpx": logging.WARNING,
    "httpcore": logging.WARNING,
    "filelock": logging.WARNING,
    "transformers": logging.ERROR,
    "sentence_transformers": logging.WARNING,
    "chromadb": logging.WARNING,
    "matplotlib": logging.WARNING,
    "PIL": logging.WARNING,
    "groq": logging.WARNING,
    "asyncio": logging.WARNING,
    "datasets": logging.WARNING,
    # Journal interne du décodeur : « EOS token detected » à chaque synthèse.
    # Utile en DEBUG, bruit sinon.
    "t3": logging.WARNING,
    "huggingface_hub": logging.WARNING,
    "numba": logging.WARNING,
}

_COULEURS = {
    logging.DEBUG:   "\033[2;37m",     # gris estompé
    logging.INFO:    "",
    logging.WARNING: "\033[1;33m",
    logging.ERROR:   "\033[1;31m",
    logging.CRITICAL: "\033[1;37;41m",
}
_FIN = "\033[0m"


class _Filtre(logging.Filter):
    """Attache l'identifiant de requête courant à chaque enregistrement."""

    def filter(self, record: logging.LogRecord) -> bool:
        # Certaines bibliothèques journalisent par `logging.info(...)` au niveau
        # module, donc directement sur le logger racine — « input frame rate=25 »
        # au chargement du vocodeur. On ne peut pas les museler par leur nom
        # sans museler le projet entier : `logging.getLogger("root")` RETOURNE
        # le logger racine, et le baisser coupe toute la journalisation. On
        # écarte donc ces enregistrements ici, sauf en DEBUG où l'on veut tout.
        if record.name == "root" and record.levelno < logging.WARNING:
            if logging.getLogger().level > logging.DEBUG:
                return False

        record.requete = _requete.get()
        # Le nom complet ('tts_Ooleil.tts') tiendrait toute la colonne : on ne
        # garde que le dernier segment, qui est celui qu'on cherche des yeux.
        #
        # Sauf quand ce segment est générique : uvicorn journalise son démarrage
        # sous 'uvicorn.error', ce qui affichait « error » en face de « Started
        # server process » — un démarrage normal avait l'air d'une panne.
        parts = record.name.split(".")
        nom = parts[-1]
        if nom in _SEGMENTS_GENERIQUES and len(parts) > 1:
            nom = parts[-2]
        record.composant = nom[:12]
        return True


class _Format(logging.Formatter):
    """Format compact, avec couleur seulement si un humain regarde."""

    def __init__(self, couleur: bool):
        super().__init__(datefmt="%H:%M:%S")
        self.couleur = couleur

    def format(self, record: logging.LogRecord) -> str:
        heure = f"{self.formatTime(record, self.datefmt)}.{int(record.msecs):03d}"
        ligne = (f"{heure}  {record.levelname:<7} {record.requete}  "
                 f"{record.composant:<12} {record.getMessage()}")
        if record.exc_info:
            ligne += "\n" + self.formatException(record.exc_info)
        if self.couleur:
            teinte = _COULEURS.get(record.levelno, "")
            if teinte:
                ligne = f"{teinte}{ligne}{_FIN}"
        return ligne


_configure = False


def configurer(niveau: str | None = None, fichier: str | None = None) -> None:
    """Installe la journalisation. Idempotent : un second appel ne fait rien.

    Appelé explicitement au démarrage du serveur, et implicitement au premier
    `journal(...)` pour que les scripts et les tests aient une sortie correcte
    sans avoir à y penser.
    """
    global _configure
    if _configure:
        return
    _configure = True

    niveau_txt = (niveau or os.getenv("LOG_LEVEL", "INFO")).upper()
    niveau_log = _NIVEAUX.get(niveau_txt, logging.INFO)

    choix = (os.getenv("LOG_COULEUR", "auto") or "auto").lower()
    couleur = sys.stderr.isatty() if choix == "auto" else choix in ("1", "oui", "true")

    racine = logging.getLogger()
    racine.setLevel(niveau_log)
    for ancien in list(racine.handlers):
        racine.removeHandler(ancien)

    # Le terminal reçoit stderr : la sortie standard reste libre pour les
    # scripts qui produisent des données, sans que les traces s'y mêlent.
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(_Format(couleur))
    console.addFilter(_Filtre())
    racine.addHandler(console)

    chemin = fichier if fichier is not None else os.getenv("LOG_FILE", "")
    if chemin:
        Path(chemin).parent.mkdir(parents=True, exist_ok=True)
        # Rotation : une borne qui tourne des semaines ne doit pas remplir son
        # disque avec ses propres traces.
        from logging.handlers import RotatingFileHandler
        fic = RotatingFileHandler(chemin, maxBytes=5_000_000, backupCount=3,
                                  encoding="utf-8")
        fic.setFormatter(_Format(couleur=False))   # pas d'échappements dans un fichier
        fic.addFilter(_Filtre())
        racine.addHandler(fic)

    for nom, plancher in _BAVARDES.items():
        logging.getLogger(nom).setLevel(max(plancher, niveau_log))

    # Les avertissements Python (torch, transformers…) passent par le journal
    # au lieu d'écrire directement sur stderr : ils deviennent filtrables et
    # horodatés comme le reste.
    logging.captureWarnings(True)
    logging.getLogger("py.warnings").setLevel(
        logging.DEBUG if niveau_log <= logging.DEBUG else logging.ERROR)
    if niveau_log > logging.DEBUG:
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)


def journal(composant: str) -> logging.Logger:
    """Journal d'un composant : 'tts', 'nllb', 'retrieval'…"""
    configurer()
    return logging.getLogger(composant)


# ── Contexte de requête ───────────────────────────────────────────────────

def nouvel_identifiant() -> str:
    """Quatre caractères, assez pour distinguer les requêtes d'une session."""
    return uuid.uuid4().hex[:4]


class contexte_requete:
    """Marque toutes les lignes émises dans ce bloc du même identifiant.

    S'utilise aussi bien en synchrone qu'en asynchrone. Attention : un fil
    lancé par `run_in_executor` n'hérite PAS du contexte de l'appelant — il
    faut rouvrir le bloc à l'intérieur du fil, ou lui passer l'identifiant.
    """

    def __init__(self, identifiant: str | None = None):
        self.identifiant = identifiant or nouvel_identifiant()
        self._jeton = None

    def __enter__(self) -> str:
        self._jeton = _requete.set(self.identifiant)
        return self.identifiant

    def __exit__(self, *_) -> None:
        if self._jeton is not None:
            _requete.reset(self._jeton)


def identifiant_courant() -> str:
    return _requete.get()
