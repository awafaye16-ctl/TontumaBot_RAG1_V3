"""STT V3 — Transcription audio wolof.

Modèle retenu après benchmark : M9and2M/whisper-small-wolof
  → Fine-tuné spécifiquement sur le wolof, 242M paramètres.
  → Chargé depuis le dossier local `wolof-whisper-small-lora/` (déjà cloné en V3/)
    OU téléchargé depuis HuggingFace Hub si le dossier local est absent.

Le modèle est chargé une seule fois (lazy loading) et reste en mémoire.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import settings  # noqa: E402

_model     = None
_processor = None
_SOURCE    = None

# Identifiant Hub en cas de téléchargement automatique
_HUB_MODEL_ID = "M9and2M/whisper-small-wolof"


def load_model():
    """Charge le modèle STT une seule fois.

    Priorité :
      1. Dossier local STT_MODEL_PATH (wolof-whisper-small-lora/)
      2. Téléchargement automatique depuis Hub (M9and2M/whisper-small-wolof)
    """
    global _model, _processor, _SOURCE
    if _model is not None:
        return _model, _processor

    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    local_path = settings.STT_MODEL_PATH
    if local_path and os.path.isdir(local_path):
        print(f"[STT] Chargement depuis dossier local : {local_path}")
        src = local_path
    else:
        print(f"[STT] Dossier local introuvable ({local_path}).")
        print(f"[STT] Téléchargement depuis Hub : {_HUB_MODEL_ID} ...")
        src = _HUB_MODEL_ID

    _processor = WhisperProcessor.from_pretrained(src)
    _model     = WhisperForConditionalGeneration.from_pretrained(src)
    _SOURCE    = src
    print(f"[STT] Modèle prêt ({src}).")
    return _model, _processor


def transcribe(audio_path: str, language: str = None) -> str:
    """Transcrit un fichier audio en texte.

    audio_path : chemin vers le fichier audio (wav, mp3, m4a, webm…)
    language   : code langue facultatif ('wo' pour forcer le wolof)

    Utilise librosa pour le chargement audio (supporte tous les formats
    grâce à soundfile/audioread) et le processor Whisper pour les features.
    """
    import torch
    import librosa

    model, processor = load_model()

    # Chargement et resample à 16 kHz (format attendu par Whisper)
    audio, _ = librosa.load(audio_path, sr=16000, mono=True)

    # Extraction des features d'entrée
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features

    # Génération avec langue forcée si fournie
    gen_kwargs = {}
    if language:
        try:
            forced_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")
            gen_kwargs["forced_decoder_ids"] = forced_ids
        except Exception:
            pass  # si la langue n'est pas reconnue, on laisse Whisper détecter

    with torch.no_grad():
        predicted_ids = model.generate(input_features, **gen_kwargs)

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription.strip()


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) < 2:
        print("Usage : python stt.py <audio_path> [language]")
        _sys.exit(1)
    audio = _sys.argv[1]
    lang  = _sys.argv[2] if len(_sys.argv) > 2 else None
    text  = transcribe(audio, language=lang)
    print(f"[STT] Transcription : {text}")
