"""Traduction bidirectionnelle Wolof ↔ Français — V3.

Utilise le même modèle bilalfaye/nllb-200-distilled-600M-wo-fr-en
pour les deux directions (WO→FR et FR→WO).

Le modèle est chargé paresseusement (au premier appel) et
reste en mémoire pour les appels suivants.
"""
import re
import time
import torch
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer

# Codes de langue NLLB (format : iso_Script)
WOLOF  = "wol_Latn"
FRENCH = "fra_Latn"

# ── Modèle unique (les deux directions) ──────────────────────────────────
from config import settings as _settings

from journal import journal

_log = journal("nllb")
_CHECKPOINT = _settings.NLLB_WO_FR_MODEL  # bilalfaye/nllb-200-distilled-600M-wo-fr-en

_tok   = None
_model = None

# Backend résolu à la première utilisation, pas à l'import : la détection
# demande torch, et un import précoce fige le choix avant que .env soit lu.
_device = None


def _get_device() -> str:
    global _device
    if _device is None:
        import device as _dev
        _device = _dev.resolve("nllb")
    return _device


def _load_model():
    """Charge le modèle NLLB une seule fois, sur le backend résolu."""
    global _tok, _model
    if _model is not None:
        return
    import device as _dev
    dev = _get_device()
    _log.info(f"Chargement ({_CHECKPOINT}) sur {dev.upper()}...")
    _tok   = NllbTokenizer.from_pretrained(_CHECKPOINT)
    _model = AutoModelForSeq2SeqLM.from_pretrained(
        _CHECKPOINT, torch_dtype=_dev.dtype("nllb")
    ).to(dev)
    _model.eval()
    _log.info("Chargé.")
# =============================================================================
#  Découpage en phrases
# =============================================================================
#  Traduire une réponse entière d'un bloc oblige le décodeur à générer ~100
#  jetons EN SÉRIE, chacun attendant le précédent. Découpée en phrases et
#  passée en lot, la même réponse se décode en parallèle et chaque élément ne
#  fait plus qu'une vingtaine de jetons : c'est la longueur de la plus longue
#  phrase qui compte, plus la somme.
#
#  Le séparateur est capturé pour être restitué tel quel : sauts de ligne et
#  espacement d'origine survivent au recollage.
#
#  Les listes numérotées (« 1. Présentez... ») produisent un segment « 1. »
#  isolé. Plutôt que d'interdire la coupure après un chiffre — ce qui souderait
#  aussi les vraies fins de phrase comme « ... au guichet numéro 4. » à la
#  phrase suivante — on découpe normalement et on laisse le marqueur de côté :
#  il n'a rien à traduire et se recolle intact.

_SEPARATEUR = re.compile(r"(\s*\n+\s*|(?<=[.!?…;])\s+)")

# Un segment qui n'est qu'un marqueur de liste : « 1. », « 2) », « 3- »
_MARQUEUR = re.compile(r"^\s*\d+\s*[.)\-]\s*$")

# Au-delà, le lot consomme trop de mémoire sur MPS/CPU sans gagner en vitesse :
# les phrases restantes partent dans un lot suivant.
_TAILLE_LOT = 8


def _segmenter(texte: str) -> tuple[list[str], list[int]]:
    """Découpe en morceaux alternés (texte, séparateur, texte, ...).

    Retourne les morceaux et les indices de ceux qui sont à traduire.
    """
    morceaux = _SEPARATEUR.split(texte)
    a_traduire = [i for i, m in enumerate(morceaux)
                  if i % 2 == 0 and m and m.strip()
                  and not _MARQUEUR.match(m)]
    return morceaux, a_traduire


def _traduire_lot(phrases: list[str], src_lang: str, tgt_lang: str) -> list[str]:
    """Traduit une liste de phrases en un seul appel au décodeur."""
    if not phrases:
        return []

    _tok.src_lang = src_lang
    inputs = _tok(
        phrases,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(_get_device())

    # Budget de génération calé sur la plus longue phrase du lot plutôt que sur
    # un plafond fixe : l'ancien `max_new_tokens=128` tronquait silencieusement
    # les réponses longues traduites d'un bloc.
    longueur = int(inputs["input_ids"].shape[-1])
    budget   = min(512, max(64, int(longueur * 2) + 16))

    with torch.no_grad():
        tokens = _model.generate(
            **inputs,
            forced_bos_token_id=_tok.convert_tokens_to_ids(tgt_lang),
            max_new_tokens=budget,
            num_beams=_settings.NLLB_NUM_BEAMS,
            early_stopping=True,
        )
    return _tok.batch_decode(tokens, skip_special_tokens=True)


def _translate(text: str, src_lang: str, tgt_lang: str) -> tuple[str, float]:
    """Traduit `text` de src_lang vers tgt_lang, phrase par phrase en lot."""
    _load_model()
    t0 = time.time()

    morceaux, a_traduire = _segmenter(text)
    if not a_traduire:
        return text, round(time.time() - t0, 2)

    phrases = [morceaux[i] for i in a_traduire]
    traduites: list[str] = []
    for debut in range(0, len(phrases), _TAILLE_LOT):
        traduites += _traduire_lot(phrases[debut:debut + _TAILLE_LOT],
                                   src_lang, tgt_lang)

    for indice, traduite in zip(a_traduire, traduites):
        morceaux[indice] = traduite

    result  = "".join(morceaux)
    elapsed = round(time.time() - t0, 2)
    return result, elapsed


def device() -> str:
    """Backend effectivement utilisé — pour /health et les traces."""
    return _get_device()


def wolof_to_french(text: str) -> tuple[str, float]:
    """Traduit du wolof vers le français."""
    return _translate(text, src_lang=WOLOF, tgt_lang=FRENCH)


def french_to_wolof(text: str) -> tuple[str, float]:
    """Traduit du français vers le wolof."""
    return _translate(text, src_lang=FRENCH, tgt_lang=WOLOF)


if __name__ == "__main__":
    print("=== Test WO→FR ===")
    tr, d = wolof_to_french("dama beug wout kayitu juddu?")
    print(f"  WO : dama beug wout kayitu juddu?")
    print(f"  FR : {tr}  ({d}s)\n")
    print("=== Test FR→WO ===")
    tr, d = french_to_wolof("Comment obtenir un extrait de naissance ?")
    print(f"  FR : Comment obtenir un extrait de naissance ?")
    print(f"  WO : {tr}  ({d}s)")