"""Configuration V3 — chargée depuis .env.

Modèles sélectionnés après benchmark :
  - WO→FR : bilalfaye/nllb-200-distilled-600M-wo-fr-en
  - FR→WO : Lahad/nllb200-francais-wolof
  - STT    : M9and2M/whisper-small-wolof  (local : wolof-whisper-small-lora/)
  - TTS    : Oolel-Voices (soynade-research/Oolel-Voices)
  - LLM    : Qwen/Qwen2.5-7B-Instruct (local 4bit) ou Groq/Gemini (API)
"""
import os
from pathlib import Path

# ── Désactiver TensorFlow/Keras avant tout import ─────────────────────────
# sentence_transformers déclenche keras qui charge TF (~20s) même si on
# n'utilise pas TF. Ces variables l'empêchent de se charger.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")          # pas de GPU TF
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
# Force sentence_transformers à utiliser PyTorch uniquement
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")   # évite les warnings HF

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"


def _load_env():
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_env()


class Settings:
    BASE_DIR = BASE_DIR

    # ── LLM ──────────────────────────────────────────────────────────────
    GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
    GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
    LLM_PROVIDER    = os.getenv("LLM_PROVIDER", "groq")
    # Modèle local (utilisé si LLM_PROVIDER == "local")
    LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    LOCAL_LLM_QUANT = os.getenv("LOCAL_LLM_QUANT", "4bit")  # 4bit | 8bit | fp16

    # ── Traduction — deux modèles distincts (résultat benchmark) ─────────
    # WO→FR : bilalfaye fine-tuné wolof-français-anglais
    NLLB_WO_FR_MODEL = os.getenv(
        "NLLB_WO_FR_MODEL",
        str(BASE_DIR / "src" / "Wo_fr_bilalfayenllb-200-distilled-600M-wo-fr-en")
        if (BASE_DIR / "src" / "Wo_fr_bilalfayenllb-200-distilled-600M-wo-fr-en" / "config.json").exists()
        else "bilalfaye/nllb-200-distilled-600M-wo-fr-en"
    )
    # FR→WO : Lahad fine-tuné français→wolof
    NLLB_FR_WO_MODEL = os.getenv(
        "NLLB_FR_WO_MODEL",
        str(BASE_DIR / "src" / "Fr-Wo_Lahadnllb200-francais-wolof")
        if (BASE_DIR / "src" / "Fr-Wo_Lahadnllb200-francais-wolof" / "config.json").exists()
        else "Lahad/nllb200-francais-wolof"
    )

    # ── STT ───────────────────────────────────────────────────────────────
    # Priorité : dossier local src/stt_wolof-whisper-small-lora/
    #            puis wolof-whisper-small-lora/ (déjà cloné en V3/)
    #            puis Hub M9and2M/whisper-small-wolof

    @staticmethod
    def _stt_local_valid(path: "Path") -> bool:
        """Vérifie que le dossier contient config.json ET les poids du modèle."""
        if not (path / "config.json").exists():
            return False
        # Au moins un fichier de poids doit être présent
        weight_files = [
            "pytorch_model.bin",
            "model.safetensors",
            "tf_model.h5",
            "model.ckpt.index",
            "flax_model.msgpack",
        ]
        return any((path / w).exists() for w in weight_files)

    @staticmethod
    def _resolve_stt_path() -> str:
        candidates = [
            BASE_DIR / "src" / "stt_wolof-whisper-small-lora",
            BASE_DIR / "wolof-whisper-small-lora",
        ]
        for c in candidates:
            if (c / "config.json").exists():
                return str(c)
        return "M9and2M/whisper-small-wolof"  # téléchargement Hub

    STT_MODEL_PATH = os.getenv(
        "STT_MODEL_PATH",
        str(BASE_DIR / "src" / "stt_wolof-whisper-small-lora")
        if (BASE_DIR / "src" / "stt_wolof-whisper-small-lora" / "config.json").exists()
        and any((BASE_DIR / "src" / "stt_wolof-whisper-small-lora" / w).exists()
                for w in ["pytorch_model.bin", "model.safetensors"])
        else str(BASE_DIR / "wolof-whisper-small-lora")
        if (BASE_DIR / "wolof-whisper-small-lora" / "config.json").exists()
        and any((BASE_DIR / "wolof-whisper-small-lora" / w).exists()
                for w in ["pytorch_model.bin", "model.safetensors"])
        else "M9and2M/whisper-small-wolof"
    )
    STT_LANGUAGE = os.getenv("STT_LANGUAGE", "")

    # ── Embeddings / Vectorstore ──────────────────────────────────────────
    EMBED_MODEL = os.getenv(
        "EMBED_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # ── TTS ───────────────────────────────────────────────────────────────
    # Moteur par défaut : oolel (meilleur qualité selon benchmark)
    TTS_ENGINE       = os.getenv("TTS_ENGINE", "oolel")
    OOLEL_TTS_REPO   = os.getenv("OOLEL_TTS_REPO", "soynade-research/Oolel-Voices")
    # Fallback SpeechT5 si Oolel indisponible
    WOLOF_TTS_MODEL  = os.getenv("WOLOF_TTS_MODEL", "bilalfaye/speecht5_tts-wolof-v0.2")

    # ── Reranker ──────────────────────────────────────────────────────────
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    RERANKER_TOP_K = int(os.getenv("RERANKER_TOP_K", "3"))

    # ── Serveur ───────────────────────────────────────────────────────────
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))

    @property
    def llm_ready(self) -> bool:
        if self.LLM_PROVIDER == "local":
            return True  # toujours disponible si les poids sont téléchargés
        if self.LLM_PROVIDER == "gemini":
            return bool(self.GEMINI_API_KEY)
        return bool(self.GROQ_API_KEY)


settings = Settings()
