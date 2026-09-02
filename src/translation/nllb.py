"""Traduction bidirectionnelle Wolof ↔ Français — V3.

Utilise le même modèle bilalfaye/nllb-200-distilled-600M-wo-fr-en
pour les deux directions (WO→FR et FR→WO).

Le modèle est chargé paresseusement (au premier appel) et
reste en mémoire pour les appels suivants.
"""
import time
import torch
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer

# Codes de langue NLLB (format : iso_Script)
WOLOF  = "wol_Latn"
FRENCH = "fra_Latn"

# ── Modèle unique (les deux directions) ──────────────────────────────────
from config import settings as _settings
_CHECKPOINT = _settings.NLLB_WO_FR_MODEL  # bilalfaye/nllb-200-distilled-600M-wo-fr-en

_tok   = None
_model = None

_device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu")


def _load_model():
    """Charge le modèle NLLB une seule fois."""
    global _tok, _model
    if _model is not None:
        return
    print(f"[NLLB] Chargement ({_CHECKPOINT}) sur {_device.upper()}...")
    _tok   = NllbTokenizer.from_pretrained(_CHECKPOINT)
    _model = AutoModelForSeq2SeqLM.from_pretrained(_CHECKPOINT).to(_device)
    _model.eval()
    print("[NLLB] Chargé.")


def _translate(text: str, src_lang: str, tgt_lang: str) -> tuple[str, float]:
    """Traduit `text` de src_lang vers tgt_lang."""
    _load_model()
    t0 = time.time()
    _tok.src_lang = src_lang
    inputs = _tok(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(_device)
    with torch.no_grad():
        tokens = _model.generate(
            **inputs,
            forced_bos_token_id=_tok.convert_tokens_to_ids(tgt_lang),
            max_new_tokens=128,
            num_beams=_settings.NLLB_NUM_BEAMS,
            early_stopping=True,
        )
    result  = _tok.batch_decode(tokens, skip_special_tokens=True)[0]
    elapsed = round(time.time() - t0, 2)
    return result, elapsed


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
