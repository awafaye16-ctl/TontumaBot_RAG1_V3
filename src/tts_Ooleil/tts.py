"""TTS V3 — Oolel-Voices uniquement.

Modèle : soynade-research/Oolel-Voices (voice cloning, meilleure qualité)
Voice prompt : fichier audio de référence pour l'identité vocale.
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import settings  # noqa: E402

import numpy as np
import soundfile as sf

# ── État ───────────────────────────────────────────────────────────────────
_model  = None
_ready  = False
_prompt = None


def _find_cache() -> Path | None:
    """Cherche le snapshot Oolel-Voices dans le cache HuggingFace local."""
    cache = (
        Path.home()
        / ".cache" / "huggingface" / "hub"
        / "models--soynade-research--Oolel-Voices"
        / "snapshots"
    )
    if cache.exists():
        dirs = sorted(cache.iterdir())
        if dirs:
            return dirs[-1]
    return None


def _load() -> bool:
    """Charge Oolel-Voices depuis le Hub (ou le cache). Retourne True si succès."""
    global _model, _ready, _prompt
    if _ready:
        return True
    try:
        ckpt_dir = _find_cache()
        if ckpt_dir is None:
            print(f"[TTS] Téléchargement Oolel-Voices ({settings.OOLEL_TTS_REPO})...")
            from huggingface_hub import snapshot_download
            ckpt_dir = Path(snapshot_download(repo_id=settings.OOLEL_TTS_REPO))
            print(f"[TTS] Oolel-Voices téléchargé → {ckpt_dir}")

        ckpt_str = str(ckpt_dir)
        if ckpt_str not in sys.path:
            sys.path.insert(0, ckpt_str)

        print(f"[TTS] Chargement Oolel-Voices depuis {ckpt_dir}...")
        from modeling_oolel_voices import OolelVoicesForInference  # type: ignore

        import torch
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            _device = "mps"
            print("[TTS] Accélération MPS (Apple Metal) activée")
        else:
            _device = "cpu"

        _model = OolelVoicesForInference.from_pretrained(ckpt_str, device_map=_device)
        _model.eval()

        # ── Accélération vocodeur : réduire les itérations de diffusion ──
        n_cfm = getattr(settings, "TTS_N_STEPS", 10)
        if n_cfm and 1 <= n_cfm < 10:
            s3gen     = _model.s3gen
            _orig_inf = s3gen.inference

            def _fast_inference(*args, n_cfm_timesteps=None, **kwargs):
                if n_cfm_timesteps is None:
                    n_cfm_timesteps = n_cfm
                return _orig_inf(*args, n_cfm_timesteps=n_cfm_timesteps, **kwargs)

            s3gen.inference = _fast_inference
            print(f"[TTS] Vocodeur accéléré : {n_cfm} itérations (au lieu de 10)")

        # Prompt audio de référence pour le voice cloning
        candidates = [
            ckpt_dir / "8_1_c.wav",
            settings.BASE_DIR / "8_1_c.wav",
            settings.BASE_DIR.parent / "Oolel-Voices" / "8_1_c.wav",
        ]
        for c in candidates:
            if c.exists():
                _prompt = str(c)
                print(f"[TTS] Voice prompt : {_prompt}")
                break

        if _prompt is None:
            print("[TTS] ⚠️  Aucun fichier audio de référence trouvé — synthèse sans voice prompt.")

        _ready = True
        print("[TTS] Oolel-Voices prêt.")
        return True

    except Exception as e:
        print(f"[TTS] Oolel-Voices indisponible ({type(e).__name__}: {e})")
        return False


def synthesize(text: str, out_path: str) -> str:
    """Synthétise le texte avec Oolel-Voices et sauvegarde dans out_path."""
    if not _load():
        raise RuntimeError("Oolel-Voices impossible à charger")

    kwargs = {}
    if _prompt:
        kwargs["audio_prompt_path"] = _prompt

    wav = _model.generate(
        text,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        **kwargs,
    )
    audio_np = wav.squeeze(0).detach().cpu().numpy()

    # ── Réglage de la vitesse (time-stretch, préserve la hauteur) ────────
    speed = getattr(settings, "TTS_SPEED", 1.0)
    if speed and speed > 0 and abs(speed - 1.0) > 1e-3:
        import librosa
        t0 = time.perf_counter()
        # librosa.effects.time_stretch(rate>1) → ralentit
        audio_np = librosa.effects.time_stretch(audio_np, rate=speed)
        print(f"[TTS] Time-stretch rate={speed} ({time.perf_counter() - t0:.2f}s)")

    sf.write(out_path, audio_np, _model.sr, format="WAV")
    return out_path


def source() -> str:
    """Retourne le nom du moteur TTS (pour les traces)."""
    return "oolel-voices" if _ready else "indisponible"


if __name__ == "__main__":
    import sys as _sys
    txt = _sys.argv[1] if len(_sys.argv) > 1 else "Jàmm nga fanaan. Nanga def?"
    out = _sys.argv[2] if len(_sys.argv) > 2 else "test_v3.wav"
    result = synthesize(txt, out)
    print(f"[TTS] {source()} → {result}")
