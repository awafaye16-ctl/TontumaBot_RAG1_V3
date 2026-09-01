"""Traduction bidirectionnelle Wolof ↔ Français — V3.

Résultat du benchmark :
  - WO→FR : bilalfaye/nllb-200-distilled-600M-wo-fr-en
            (fine-tuné spécifiquement sur paires wolof-français-anglais,
             meilleur sur le vocabulaire administratif sénégalais)
  - FR→WO : Lahad/nllb200-francais-wolof
            (fine-tuné français→wolof, produit un wolof plus naturel
             et idiomatique que bilalfaye dans ce sens)

Les deux modèles sont chargés paresseusement (au premier appel) et
restent en mémoire pour les appels suivants.
"""
import time
import torch
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer

# Codes de langue NLLB (format : iso_Script)
WOLOF  = "wol_Latn"
FRENCH = "fra_Latn"

# ── Modèle WO→FR ──────────────────────────────────────────────────────────
# Lit depuis config (dossier local si présent, sinon Hub)
from config import settings as _settings
_WO_FR_CHECKPOINT = _settings.NLLB_WO_FR_MODEL
_FR_WO_CHECKPOINT = _settings.NLLB_FR_WO_MODEL

_tok_wo_fr   = None
_model_wo_fr = None
_tok_fr_wo   = None
_model_fr_wo = None

_device = "cuda" if torch.cuda.is_available() else "cpu"


def _load_wo_fr():
    """Charge le modèle WO→FR une seule fois."""
    global _tok_wo_fr, _model_wo_fr
    if _model_wo_fr is not None:
        return
    print(f"[NLLB] Chargement WO→FR ({_WO_FR_CHECKPOINT}) sur {_device.upper()}...")
    _tok_wo_fr   = NllbTokenizer.from_pretrained(_WO_FR_CHECKPOINT)
    _model_wo_fr = AutoModelForSeq2SeqLM.from_pretrained(_WO_FR_CHECKPOINT).to(_device)
    _model_wo_fr.eval()
    print("[NLLB] WO→FR chargé.")


def _load_fr_wo():
    """Charge le modèle FR→WO une seule fois."""
    global _tok_fr_wo, _model_fr_wo
    if _model_fr_wo is not None:
        return
    print(f"[NLLB] Chargement FR→WO ({_FR_WO_CHECKPOINT}) sur {_device.upper()}...")
    _tok_fr_wo   = NllbTokenizer.from_pretrained(_FR_WO_CHECKPOINT)
    _model_fr_wo = AutoModelForSeq2SeqLM.from_pretrained(_FR_WO_CHECKPOINT).to(_device)
    _model_fr_wo.eval()
    print("[NLLB] FR→WO chargé.")


def _translate(text: str, tokenizer, model, src_lang: str, tgt_lang: str) -> tuple[str, float]:
    """Traduit `text` avec le modèle/tokenizer fournis."""
    t0 = time.time()
    tokenizer.src_lang = src_lang
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(_device)
    with torch.no_grad():
        tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
            max_new_tokens=128,
            num_beams=4,
            early_stopping=True,
        )
    result  = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
    elapsed = round(time.time() - t0, 2)
    return result, elapsed


def wolof_to_french(text: str) -> tuple[str, float]:
    """Traduit du wolof vers le français.
    Modèle : bilalfaye/nllb-200-distilled-600M-wo-fr-en
    """
    _load_wo_fr()
    return _translate(text, _tok_wo_fr, _model_wo_fr, src_lang=WOLOF, tgt_lang=FRENCH)


def french_to_wolof(text: str) -> tuple[str, float]:
    """Traduit du français vers le wolof.
    Modèle : Lahad/nllb200-francais-wolof
    """
    _load_fr_wo()
    return _translate(text, _tok_fr_wo, _model_fr_wo, src_lang=FRENCH, tgt_lang=WOLOF)


if __name__ == "__main__":
    print("=== Test WO→FR ===")
    tr, d = wolof_to_french("dama beug wout kayitu juddu?")
    print(f"  WO : dama beug wout kayitu juddu?")
    print(f"  FR : {tr}  ({d}s)\n")

    print("=== Test FR→WO ===")
    tr, d = french_to_wolof("Comment obtenir un extrait de naissance ?")
    print(f"  FR : Comment obtenir un extrait de naissance ?")
    print(f"  WO : {tr}  ({d}s)")
