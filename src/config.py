"""Configuration V3 — chargée depuis .env.

Modèles sélectionnés après benchmark :
  - WO→FR : bilalfaye/nllb-200-distilled-600M-wo-fr-en
  - FR→WO : bilalfaye/nllb-200-distilled-600M-wo-fr-en (même modèle)
  - STT wo : soynade-research/Wolof-HuBERT-CTC (local : src/stt_wolof-hubert-ctc/)
  - STT fr : openai/whisper-large-v3-turbo
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


def _env_bool(key: str, default: bool) -> bool:
    return os.getenv(key, str(default)).strip().lower() in ("1", "true", "yes", "on")


# ── Résolution du checkpoint STT ─────────────────────────────────────────
_STT_HUB_ID  = "soynade-research/Wolof-HuBERT-CTC"
_STT_WEIGHTS = ("model.safetensors", "pytorch_model.bin")


def _stt_local_valid(path: Path) -> bool:
    """Le dossier contient-il config.json ET au moins un fichier de poids ?"""
    return ((path / "config.json").exists()
            and any((path / w).exists() for w in _STT_WEIGHTS))


def _resolve_stt_path() -> str:
    """Dossier local du modèle STT s'il est complet, sinon l'identifiant Hub."""
    local = BASE_DIR / "src" / "stt_wolof-hubert-ctc"
    return str(local) if _stt_local_valid(local) else _STT_HUB_ID


class Settings:
    BASE_DIR = BASE_DIR

    # ── LLM ──────────────────────────────────────────────────────────────
    GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
    GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
    LLM_PROVIDER    = os.getenv("LLM_PROVIDER", "groq")
    # Modèle Groq : openai/gpt-oss-120b (non-reasoning, rapide, pas de <think>)
    GROQ_MODEL      = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    GEMINI_MODEL    = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    # Modèle local (utilisé si LLM_PROVIDER == "local")
    LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    LOCAL_LLM_QUANT = os.getenv("LOCAL_LLM_QUANT", "4bit")  # 4bit | 8bit | fp16

    # ── Traduction — un seul modèle pour les deux directions ─────────────
    # bilalfaye/nllb-200-distilled-600M-wo-fr-en (WO→FR et FR→WO)
    NLLB_WO_FR_MODEL = os.getenv(
        "NLLB_WO_FR_MODEL",
        str(BASE_DIR / "src" / "Wo_fr_bilalfayenllb-200-distilled-600M-wo-fr-en")
        if (BASE_DIR / "src" / "Wo_fr_bilalfayenllb-200-distilled-600M-wo-fr-en" / "config.json").exists()
        else "bilalfaye/nllb-200-distilled-600M-wo-fr-en"
    )
    # Nombre de beams pour la traduction : 2 = qualité, 1 = greedy (plus rapide)
    NLLB_NUM_BEAMS = int(os.getenv("NLLB_NUM_BEAMS", "2"))

    # ── STT — un moteur par langue ────────────────────────────────────────
    # Wolof : dossier local src/stt_wolof-hubert-ctc/, sinon le Hub.
    # (STT_MODEL_PATH reste accepté : c'est l'ancien nom de STT_WO_MODEL.)
    STT_WO_MODEL = os.getenv("STT_WO_MODEL",
                             os.getenv("STT_MODEL_PATH", _resolve_stt_path()))

    # Français : Whisper. Mesuré sur cette machine (CPU, phrase de 6 s) :
    #   openai/whisper-small           971 Mo — 1,3 s  ← défaut
    #   openai/whisper-large-v3-turbo  1,6 Go — plus précis en audio bruité,
    #                                  mais l'encodeur traite toujours 30 s de
    #                                  signal : compter plusieurs secondes sans GPU.
    # Basculer = une variable dans .env, aucun changement de code.
    STT_FR_MODEL = os.getenv("STT_FR_MODEL", "openai/whisper-small")

    # Langue utilisée quand la requête n'en précise pas (la borne l'envoie
    # toujours, selon le bouton pressé).
    STT_LANGUAGE = os.getenv("STT_LANGUAGE", "wo") or "wo"

    # Précharger aussi le moteur français au démarrage (voir WARMUP_STT)
    STT_WARMUP_FR = _env_bool("STT_WARMUP_FR", False)

    # ── Mémoire conversationnelle ─────────────────────────────────────────
    # Tableau [{role, content}] par session. Réinitialisé dès qu'il atteint
    # MEMORY_MAX_MESSAGES entrées (10 = cinq échanges).
    MEMORY_ENABLED      = _env_bool("MEMORY_ENABLED", True)
    MEMORY_MAX_MESSAGES = int(os.getenv("MEMORY_MAX_MESSAGES", "10"))
    MEMORY_TTL_MINUTES  = int(os.getenv("MEMORY_TTL_MINUTES", "120"))

    # ── Embeddings / Vectorstore ──────────────────────────────────────────
    EMBED_MODEL = os.getenv(
        "EMBED_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # ── TTS ───────────────────────────────────────────────────────────────
    OOLEL_TTS_REPO = os.getenv("OOLEL_TTS_REPO", "soynade-research/Oolel-Voices")
    # Vitesse de lecture : 1.0 = normale, >1 ralentit (voix plus posée), <1 accélère
    TTS_SPEED = float(os.getenv("TTS_SPEED", "1.0"))
    # Itérations de diffusion du vocodeur : 10 = qualité originale, 4 = ~2.5x plus rapide
    TTS_N_STEPS = int(os.getenv("TTS_N_STEPS", "10"))

    # ── Reranker ──────────────────────────────────────────────────────────
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    RERANKER_TOP_K = int(os.getenv("RERANKER_TOP_K", "3"))

    # ── Serveur ───────────────────────────────────────────────────────────
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))

    # HTTPS (optionnel). Les navigateurs ne donnent accès au micro que dans un
    # contexte sécurisé : en HTTP l'enregistrement ne marche que sur localhost
    # (et Safari le refuse même là). Renseigner ces deux chemins sert donc à
    # utiliser le micro depuis un téléphone ou une autre machine du réseau.
    SSL_CERTFILE = os.getenv("SSL_CERTFILE", "")
    SSL_KEYFILE  = os.getenv("SSL_KEYFILE", "")

    @property
    def ssl_enabled(self) -> bool:
        return bool(self.SSL_CERTFILE and self.SSL_KEYFILE)

    # ── Warm-up au démarrage ──────────────────────────────────────────────
    # Précharge les modèles au lancement de l'app pour éviter la latence de
    # chargement sur la première requête. Chaque modèle peut être désactivé.
    WARMUP_ON_START = _env_bool("WARMUP_ON_START", True)
    WARMUP_STT      = _env_bool("WARMUP_STT", True)
    WARMUP_TTS      = _env_bool("WARMUP_TTS", True)

    @property
    def llm_ready(self) -> bool:
        if self.LLM_PROVIDER == "local":
            return True  # toujours disponible si les poids sont téléchargés
        if self.LLM_PROVIDER == "gemini":
            return bool(self.GEMINI_API_KEY)
        return bool(self.GROQ_API_KEY)


settings = Settings()
