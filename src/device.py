"""Sélection centralisée du backend de calcul — CPU / CUDA / MPS.

Chaque composant (STT, NLLB, TTS, embedder, reranker, LLM local) passait
jusqu'ici par sa propre logique de détection, avec des résultats incohérents :
NLLB et le TTS basculaient sur MPS, le STT restait sur CPU, et le reranker
laissait sentence-transformers décider seul. Ce module donne une réponse unique
et surchargeable.

Ordre de résolution, du plus spécifique au plus général :

  1. <COMPOSANT>_DEVICE   ex. STT_DEVICE=cpu    — force un seul composant
  2. DEVICE               ex. DEVICE=cpu        — force tout le pipeline
  3. détection auto       cuda > mps > cpu

Un device demandé mais indisponible ne fait pas échouer le démarrage : on
retombe sur le meilleur disponible en le signalant, parce qu'une borne qui
démarre lentement reste préférable à une borne qui ne démarre pas.

Note MPS : `torch.backends.mps.is_available()` ne suffit pas — un binaire
compilé sans support Metal renvoie True à `is_built()` sur une machine où le
runtime est absent. Les deux vérifications sont donc conservées.
"""
import os

from journal import journal

_log = journal("device")

_CHOIX_VALIDES = ("cuda", "mps", "cpu")

# Résolutions déjà calculées : la détection ne coûte presque rien, mais on veut
# surtout que le message d'information ne soit imprimé qu'une fois par composant.
_cache: dict[str, str] = {}


def _disponible(nom: str) -> bool:
    """Le backend `nom` est-il réellement utilisable sur cette machine ?"""
    import torch

    if nom == "cpu":
        return True
    if nom == "cuda":
        return torch.cuda.is_available()
    if nom == "mps":
        return (torch.backends.mps.is_available()
                and torch.backends.mps.is_built())
    return False


def _auto() -> str:
    """Meilleur backend disponible, par ordre de performance décroissante."""
    for nom in _CHOIX_VALIDES:
        if _disponible(nom):
            return nom
    return "cpu"


def _demande(composant: str | None) -> str | None:
    """Valeur de <COMPOSANT>_DEVICE, sinon DEVICE. None si aucune, ou 'auto'."""
    cles = [f"{composant.upper()}_DEVICE"] if composant else []
    cles.append("DEVICE")
    for cle in cles:
        val = (os.getenv(cle) or "").strip().lower()
        if val and val != "auto":
            return val
    return None


def resolve(composant: str | None = None, *, verbeux: bool = True) -> str:
    """Retourne 'cuda', 'mps' ou 'cpu' pour ce composant.

    `composant` est un nom court ('stt', 'nllb', 'tts', 'embedder',
    'reranker', 'llm') qui active la surcharge <COMPOSANT>_DEVICE.
    """
    cle_cache = composant or "_global"
    if cle_cache in _cache:
        return _cache[cle_cache]

    voulu = _demande(composant)
    # Le composant concerné, sans crochets : la colonne du journal porte déjà
    # « device », on n'y ajoute que la précision utile (nllb, tts, embedder…).
    etiquette = composant or "global"

    if voulu is None:
        choisi = _auto()
    elif voulu not in _CHOIX_VALIDES:
        _log.warning("%s valeur inconnue '%s' (attendu : %s, ou auto) — "
                     "détection automatique.", etiquette, voulu, ', '.join(_CHOIX_VALIDES))
        choisi = _auto()
    elif not _disponible(voulu):
        choisi = _auto()
        _log.warning("%s '%s' demandé mais indisponible sur cette machine — "
                     "repli sur '%s'.", etiquette, voulu, choisi)
    else:
        choisi = voulu

    if verbeux:
        origine = "imposé" if voulu and _disponible(voulu) else "auto"
        _log.info("%s → %s (%s)", etiquette, choisi.upper(), origine)
    _cache[cle_cache] = choisi
    return choisi


def torch_device(composant: str | None = None):
    """Le même choix, sous forme de `torch.device` — pour `.to(...)`."""
    import torch
    return torch.device(resolve(composant))


def pipeline_device(composant: str | None = None):
    """Valeur à passer au paramètre `device` d'un pipeline transformers.

    Les pipelines acceptent un `torch.device` ; l'ancien encodage entier
    (0 = premier GPU, -1 = CPU) ne sait pas exprimer MPS.
    """
    return torch_device(composant)


def dtype(composant: str | None = None):
    """Précision de calcul adaptée au backend.

    fp16 sur CUDA : environ deux fois plus rapide, sans perte perceptible en
    traduction et en ASR. Sur MPS et CPU on reste en fp32 — le fp16 y est soit
    instable (certains noyaux Metal), soit plus lent (pas d'unité fp16 CPU).
    """
    import torch
    return torch.float16 if resolve(composant) == "cuda" else torch.float32


def infos() -> dict:
    """État des backends — pour /health et le diagnostic."""
    import torch
    return {
        "torch":       torch.__version__,
        "auto":        _auto(),
        "cuda":        _disponible("cuda"),
        "mps":         _disponible("mps"),
        "override":    os.getenv("DEVICE", "auto"),
        "par_composant": dict(_cache),
    }


if __name__ == "__main__":
    import torch
    _log.info(f"torch {torch.__version__}")
    for nom in _CHOIX_VALIDES:
        _log.error(f"  {nom:5} : {'disponible' if _disponible(nom) else 'indisponible'}")
    _log.info(f"\nauto → {_auto()}\n")
    for c in ("stt", "nllb", "tts", "embedder", "reranker", "llm"):
        _log.info(f"  {c:9} → {resolve(c, verbeux=False)}")