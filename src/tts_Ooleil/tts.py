"""TTS V3 — Oolel-Voices (principal) + SpeechT5 wolof (fallback) + edge-tts (dernier recours).

Architecture de fallback :
  1. Oolel-Voices  (soynade-research/Oolel-Voices) — voice cloning, meilleure qualité
     → chargé depuis HuggingFace Hub (snapshot_download) si pas en cache local
  2. SpeechT5 wolof (bilalfaye/speecht5_tts-wolof-v0.2) — si Oolel indisponible
  3. edge-tts FR  (fr-FR-DeniseNeural)               — dernier recours sans modèle

Le moteur actif est contrôlé par TTS_ENGINE dans .env :
  TTS_ENGINE=oolel      → essaie Oolel, puis SpeechT5, puis edge-tts
  TTS_ENGINE=speecht5   → essaie SpeechT5, puis edge-tts (ignore Oolel)
  TTS_ENGINE=edge       → edge-tts directement

Les modèles sont récupérés directement depuis HuggingFace Hub au premier appel.
"""
import asyncio
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import settings  # noqa: E402

import edge_tts
import numpy as np
import soundfile as sf
import torch

VOICE_FR  = "fr-FR-DeniseNeural"
MAX_CHARS = 80          # taille max d'un segment SpeechT5
RATE_S5   = 16000       # sample rate SpeechT5

# ── État Oolel-Voices ──────────────────────────────────────────────────────
_oolel_model  = None
_oolel_ready  = False
_oolel_prompt = None   # chemin vers le fichier audio de référence (voice prompt)

# ── État SpeechT5 ─────────────────────────────────────────────────────────
_s5_model     = None
_s5_processor = None
_s5_vocoder   = None
_s5_spk_emb   = None
_s5_ready     = False


# =============================================================================
#  Oolel-Voices
# =============================================================================

def _find_oolel_cache() -> Path | None:
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


def _load_oolel() -> bool:
    """Charge Oolel-Voices depuis le Hub (ou le cache). Retourne True si succès."""
    global _oolel_model, _oolel_ready, _oolel_prompt
    if _oolel_ready:
        return True
    try:
        ckpt_dir = _find_oolel_cache()
        if ckpt_dir is None:
            # Téléchargement depuis HuggingFace Hub
            print(f"[TTS] Téléchargement Oolel-Voices ({settings.OOLEL_TTS_REPO})...")
            from huggingface_hub import snapshot_download
            ckpt_dir = Path(snapshot_download(repo_id=settings.OOLEL_TTS_REPO))
            print(f"[TTS] Oolel-Voices téléchargé → {ckpt_dir}")

        # Ajoute le dossier au sys.path pour importer modeling_oolel_voices
        ckpt_str = str(ckpt_dir)
        if ckpt_str not in sys.path:
            sys.path.insert(0, ckpt_str)

        print(f"[TTS] Chargement Oolel-Voices depuis {ckpt_dir}...")
        from modeling_oolel_voices import OolelVoicesForInference  # type: ignore

        _oolel_model = OolelVoicesForInference.from_pretrained(ckpt_str, device_map="cpu")
        _oolel_model.eval()

        # Prompt audio de référence pour le voice cloning
        # Cherche d'abord dans le snapshot, puis dans le dossier Oolel-Voices local
        candidates = [
            ckpt_dir / "8_1_c.wav",
            settings.BASE_DIR.parent / "Oolel-Voices" / "8_1_c.wav",
        ]
        for c in candidates:
            if c.exists():
                _oolel_prompt = str(c)
                print(f"[TTS] Voice prompt : {_oolel_prompt}")
                break

        if _oolel_prompt is None:
            print("[TTS] ⚠️  Aucun fichier audio de référence trouvé — synthèse sans voice prompt.")

        _oolel_ready = True
        print("[TTS] Oolel-Voices prêt.")
        return True

    except Exception as e:
        print(f"[TTS] Oolel-Voices indisponible ({e}) — passage au fallback SpeechT5.")
        return False


def _synth_oolel(text: str, out_path: str) -> bool:
    """Synthétise avec Oolel-Voices. Retourne True si succès."""
    if not _load_oolel():
        return False
    try:
        kwargs = {}
        if _oolel_prompt:
            kwargs["audio_prompt_path"] = _oolel_prompt

        wav = _oolel_model.generate(
            text,
            exaggeration=0.5,
            cfg_weight=0.5,
            temperature=0.8,
            **kwargs,
        )
        audio_np = wav.squeeze(0).detach().cpu().numpy()
        sf.write(out_path, audio_np, _oolel_model.sr, format="WAV")
        return True
    except Exception as e:
        print(f"[TTS] Oolel-Voices — échec synthèse ({e})")
        return False


# =============================================================================
#  SpeechT5 wolof (fallback)
# =============================================================================

def _load_speecht5() -> bool:
    """Charge SpeechT5 depuis HuggingFace Hub. Retourne True si succès."""
    global _s5_model, _s5_processor, _s5_vocoder, _s5_spk_emb, _s5_ready
    if _s5_ready:
        return True
    checkpoint = settings.WOLOF_TTS_MODEL
    if not checkpoint:
        return False
    try:
        from transformers import (
            SpeechT5ForTextToSpeech,
            SpeechT5HifiGan,
            SpeechT5Processor,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[TTS] Chargement SpeechT5 ({checkpoint}) sur {device.upper()}...")
        _s5_processor = SpeechT5Processor.from_pretrained(checkpoint)
        _s5_model     = SpeechT5ForTextToSpeech.from_pretrained(checkpoint).to(device)
        _s5_vocoder   = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

        # Speaker embedding — xvector CMU Arctic depuis HuggingFace Hub
        try:
            from datasets import load_dataset
            emb_ds   = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
            _s5_spk_emb = torch.tensor(emb_ds[7306]["xvector"]).unsqueeze(0).to(device)
            print("[TTS] Speaker embedding xvector chargé (CMU Arctic).")
        except Exception as e_emb:
            # Fallback : charger depuis le fichier .npy local si présent
            npy = os.path.join(
                os.path.dirname(__file__), "..", "..", "data", "speaker_embedding.npy"
            )
            if os.path.exists(npy):
                emb = np.load(npy)
                _s5_spk_emb = torch.tensor(emb).unsqueeze(0).to(device)
                print(f"[TTS] Speaker embedding chargé depuis {npy}.")
            else:
                _s5_spk_emb = torch.randn(1, 512).to(device)
                print(f"[TTS] ⚠️  Speaker embedding aléatoire ({e_emb}).")

        _s5_ready = True
        print("[TTS] SpeechT5 prêt.")
        return True
    except Exception as e:
        print(f"[TTS] SpeechT5 indisponible ({e})")
        return False


def _split_text(text: str) -> list[str]:
    """Découpe le texte en segments ≤ MAX_CHARS sur les frontières de phrase."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?…])\s+", text)
    segments: list[str] = []
    for p in parts:
        while len(p) > MAX_CHARS:
            cut = p.rfind(" ", 0, MAX_CHARS)
            if cut < MAX_CHARS // 2:
                cut = MAX_CHARS
            segments.append(p[:cut].strip())
            p = p[cut:].strip()
        if p:
            segments.append(p)
    return segments


def _synth_speecht5(text: str, out_path: str) -> bool:
    """Synthétise avec SpeechT5. Retourne True si succès."""
    if not _load_speecht5():
        return False
    try:
        segments = _split_text(text)
        if not segments:
            return False
        chunks = []
        for seg in segments:
            inputs = _s5_processor(text=seg, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(_s5_model.device) for k, v in inputs.items()}
            with torch.no_grad():
                wav = _s5_model.generate(
                    inputs["input_ids"],
                    speaker_embeddings=_s5_spk_emb,
                    vocoder=_s5_vocoder,
                    num_beams=5,
                    temperature=0.6,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.5,
                ).squeeze().cpu().numpy()
            chunks.append(wav)
        audio = np.concatenate(chunks)
        # Normalisation volume (pic cible 0.9, gain max 6×)
        peak = np.abs(audio).max()
        if peak > 1e-6:
            audio = audio * min(0.9 / peak, 6.0)
        sf.write(out_path, audio, RATE_S5, format="WAV")
        return True
    except Exception as e:
        print(f"[TTS] SpeechT5 — échec synthèse ({e})")
        return False


# =============================================================================
#  edge-tts (dernier recours)
# =============================================================================

async def _synth_edge_async(text: str, out_path: str) -> None:
    communicate = edge_tts.Communicate(text, VOICE_FR)
    await communicate.save(out_path)


def _synth_edge(text: str, out_path: str) -> None:
    asyncio.run(_synth_edge_async(text, out_path))


# =============================================================================
#  API publique
# =============================================================================

def synthesize(text: str, out_path: str, engine: str = None) -> str:
    """Génère la voix et sauvegarde dans out_path.

    engine : 'oolel' | 'speecht5' | 'edge' | None → utilise TTS_ENGINE du .env
    Retourne le chemin du fichier audio généré.
    """
    engine = (engine or settings.TTS_ENGINE).lower()

    if engine == "oolel":
        if _synth_oolel(text, out_path):
            return out_path
        # Oolel a échoué → fallback SpeechT5
        print("[TTS] Fallback SpeechT5...")
        if _synth_speecht5(text, out_path):
            return out_path

    elif engine == "speecht5":
        if _synth_speecht5(text, out_path):
            return out_path

    # Dernier recours : edge-tts (français)
    print("[TTS] Fallback edge-tts FR...")
    _synth_edge(text, out_path)
    return out_path


def source(engine: str = None) -> str:
    """Retourne le nom du moteur TTS effectif (pour les traces)."""
    engine = (engine or settings.TTS_ENGINE).lower()
    if engine == "oolel":
        return "oolel-voices" if _oolel_ready else (
            "speecht5-wolof" if _s5_ready else "edge-tts-fr"
        )
    if engine == "speecht5":
        return "speecht5-wolof" if _s5_ready else "edge-tts-fr"
    return "edge-tts-fr"


if __name__ == "__main__":
    import sys as _sys
    txt = _sys.argv[1] if len(_sys.argv) > 1 else "Jàmm nga fanaan. Nanga def?"
    out = _sys.argv[2] if len(_sys.argv) > 2 else "test_v3.wav"
    result = synthesize(txt, out)
    print(f"[TTS] {source()} → {result}")
