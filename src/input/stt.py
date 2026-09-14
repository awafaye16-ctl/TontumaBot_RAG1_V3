"""STT V3 — Transcription audio bilingue wolof / français.

Deux moteurs spécialisés, choisis par l'appelant (bouton de la borne) :

  wolof    soynade-research/Wolof-HuBERT-CTC
           HuBERT base fine-tuné en CTC (95M params, WER 0,357). Décodage en
           une passe, ~1,4 s pour 25 s d'audio sur CPU.

  français STT_FR_MODEL, par défaut openai/whisper-small
           Whisper encodeur-décodeur, langue forcée à `fr`. Les variantes
           « turbo » et distillées n'allègent que le décodeur : l'encodeur
           reste celui de large-v3 et traite toujours 30 s de signal, donc
           elles sont plus lentes que `small` sur des énoncés courts.

Pourquoi deux modèles plutôt qu'un Whisper multilingue : le wolof ne fait pas
partie des langues de Whisper — un modèle généraliste y produit du charabia,
alors que le CTC wolof est à la fois meilleur et plus rapide. Inversement le
CTC wolof ne sait pas transcrire le français.

Chaque moteur est chargé à la demande et reste en mémoire.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import settings  # noqa: E402

from journal import journal

_log = journal("stt")

WOLOF    = "wo"
FRANCAIS = "fr"

# Identifiant Hub du modèle wolof, si le dossier local est absent ou invalide
_HUB_WO_MODEL = "soynade-research/Wolof-HuBERT-CTC"

# Découpage des longs enregistrements. Le pipeline HF découpe en fenêtres et
# recolle les sorties (pas de mot coupé aux jointures).
_CHUNK_S  = 20
_STRIDE_S = (4, 2)

# Pipelines ASR construits à la demande, par langue
_engines: dict[str, object] = {}
_sources: dict[str, str]    = {}


def _device():
    """Backend du STT, résolu par src/device.py (surcharge : STT_DEVICE).

    L'ancienne version renvoyait -1 (CPU) dès que CUDA manquait, ce qui excluait
    MPS par principe — le commentaire invoquait des convolutions capricieuses
    dans l'extracteur de features, vrai sur les premières versions du backend
    Metal. Le support s'est stabilisé depuis ; le choix est donc laissé à la
    détection, et `STT_DEVICE=cpu` reste là pour revenir en arrière sans
    toucher au code si un checkpoint particulier pose problème.
    """
    import device as _dev
    return _dev.pipeline_device("stt")


def _is_hubert_ctc(path: str) -> bool:
    """Le dossier contient-il bien un checkpoint HuBERT-CTC ?

    Évite l'erreur obscure quand STT_WO_MODEL pointe encore sur un ancien
    checkpoint Whisper : on retombe alors proprement sur le Hub.
    """
    cfg = os.path.join(path, "config.json")
    if not os.path.isfile(cfg):
        return False
    try:
        with open(cfg, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    archs = data.get("architectures") or []
    return data.get("model_type") == "hubert" and any(a.endswith("ForCTC") for a in archs)


# =============================================================================
#  Moteur wolof — HuBERT-CTC
# =============================================================================

def _build_wolof():
    from transformers import HubertForCTC, Wav2Vec2Processor, pipeline

    local = settings.STT_WO_MODEL
    if local and os.path.isdir(local) and _is_hubert_ctc(local):
        src = local
        _log.info(f"Chargement depuis dossier local : {src}")
    else:
        if local and os.path.isdir(local):
            _log.warning(f"{local} n'est pas un checkpoint HuBERT-CTC — ignoré.")
        src = _HUB_WO_MODEL
        _log.info(f"Téléchargement depuis Hub : {src} ...")
    processor = Wav2Vec2Processor.from_pretrained(src)
    model     = HubertForCTC.from_pretrained(src)
    model.eval()

    _sources[WOLOF] = src
    return pipeline(
        task              = "automatic-speech-recognition",
        model             = model,
        tokenizer         = processor.tokenizer,
        feature_extractor = processor.feature_extractor,
        chunk_length_s    = _CHUNK_S,
        stride_length_s   = _STRIDE_S,
        device            = _device(),
    )


# =============================================================================
#  Moteur français — Whisper
# =============================================================================

def _build_francais():
    from transformers import pipeline

    src = settings.STT_FR_MODEL
    _log.info(f"Chargement de {src} ...")
    # Whisper encode toujours une fenêtre de 30 s : inutile de la fractionner
    # davantage, on aligne le découpage sur cette fenêtre native.
    asr = pipeline(
        task           = "automatic-speech-recognition",
        model          = src,
        chunk_length_s = 30,
        device         = _device(),
    )

    # Les checkpoints Whisper embarquent des `forced_decoder_ids` hérités ; ils
    # entrent en conflit avec la langue qu'on impose à chaque appel.
    try:
        asr.model.generation_config.forced_decoder_ids = None
    except AttributeError:
        pass

    _sources[FRANCAIS] = src
    return asr


_BUILDERS = {WOLOF: _build_wolof, FRANCAIS: _build_francais}


def load_model(language: str = WOLOF):
    """Charge (une seule fois) le moteur d'une langue et le retourne.

    Appelé par le warm-up au démarrage du serveur et par `transcribe`.
    """
    lang = FRANCAIS if str(language).lower().startswith("fr") else WOLOF
    if lang not in _engines:
        _engines[lang] = _BUILDERS[lang]()
        _log.info(f"Modèle prêt ({_sources.get(lang, settings.STT_FR_MODEL)}).")
    return _engines[lang]


def transcribe(audio_path: str, language: str = None) -> str:
    """Transcrit un fichier audio.

    audio_path : chemin du fichier (wav, mp3, m4a, webm, ogg…)
    language   : 'wo' (défaut) ou 'fr' — sélectionne le moteur. Sur la borne,
                 la valeur vient du bouton sur lequel l'usager a appuyé.

    Le chargement audio passe par librosa (soundfile/audioread), qui accepte
    tous les conteneurs produits par les navigateurs.
    """
    import librosa

    lang = FRANCAIS if str(language or settings.STT_LANGUAGE).lower().startswith("fr") else WOLOF

    audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    if audio.size == 0:
        return ""

    asr    = load_model(lang)
    kwargs = {"generate_kwargs": {"language": "french", "task": "transcribe"}} if lang == FRANCAIS else {}
    result = asr({"raw": audio, "sampling_rate": 16000}, **kwargs)
    return (result.get("text") or "").strip()


def source(language: str = WOLOF) -> str:
    """Identifiant du checkpoint effectivement chargé pour cette langue."""
    lang = FRANCAIS if str(language).lower().startswith("fr") else WOLOF
    return _sources.get(lang, settings.STT_FR_MODEL if lang == FRANCAIS else settings.STT_WO_MODEL)


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) < 2:
        print("Usage : python stt.py <audio_path> [wo|fr]")
        _sys.exit(1)
    lang = _sys.argv[2] if len(_sys.argv) > 2 else WOLOF
    print(f"Transcription : {transcribe(_sys.argv[1], language=lang)}")