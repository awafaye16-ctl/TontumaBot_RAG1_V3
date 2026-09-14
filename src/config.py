"""Configuration V3 — chargée depuis .env.

Modèles sélectionnés après benchmark :
  - WO→FR : bilalfaye/nllb-200-distilled-600M-wo-fr-en
  - FR→WO : bilalfaye/nllb-200-distilled-600M-wo-fr-en (même modèle)
  - STT wo : soynade-research/Wolof-HuBERT-CTC (local : src/stt_wolof-hubert-ctc/)
  - STT fr : openai/whisper-small
  - TTS    : Oolel-Voices (soynade-research/Oolel-Voices)
  - LLM    : Qwen/Qwen2.5-7B-Instruct (local 4bit) ou Groq/Gemini (API)
"""
import os
import re
from pathlib import Path

# ── Désactiver TensorFlow/Keras avant tout import ─────────────────────────
# sentence_transformers déclenche keras qui charge TF (~20s) même si on
# n'utilise pas TF. USE_TF=0 suffit à empêcher transformers de l'importer.
#
# On ne touche PAS à CUDA_VISIBLE_DEVICES ici : cette variable est lue par
# PyTorch autant que par TensorFlow. La mettre à "" pour « désactiver le GPU
# TF » rendait `torch.cuda.is_available()` faux, et faisait donc tomber NLLB,
# le STT et le LLM local sur CPU sur toute machine à GPU — sans le moindre
# message. Le choix du backend appartient désormais à src/device.py.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
# Force sentence_transformers à utiliser PyTorch uniquement
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")   # évite les warnings HF

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"


def _valeur_env(brut: str) -> str:
    """Nettoie la partie droite d'une ligne de .env.

    Le commentaire de fin de ligne doit disparaître : `.env.example` en est
    rempli, et `cp .env.example .env` — ce que font le README et demarrer.sh —
    produisait sinon une configuration qui plante à l'import, `NLLB_NUM_BEAMS`
    valant la chaîne « 2         # 2 = qualité, 1 = greedy ».

    Un « # » ne démarre un commentaire que précédé d'une espace : une clé d'API
    ou un mot de passe peut en contenir un, et le couper là silencieusement
    donnerait une panne d'authentification impossible à comprendre.

    Une valeur entre guillemets est prise telle quelle, sans interpréter ce qui
    suit — c'est la convention usuelle des fichiers .env.
    """
    brut = brut.strip()
    if brut[:1] in ("'", '"'):
        fin = brut.find(brut[0], 1)
        return brut[1:fin] if fin > 0 else brut[1:]
    coupe = re.search(r"\s#", brut)
    return brut[:coupe.start()].strip() if coupe else brut


def _load_env():
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), _valeur_env(value))


_load_env()


def _env_bool(key: str, default: bool) -> bool:
    return os.getenv(key, str(default)).strip().lower() in ("1", "true", "yes", "on")


# ── Plafond d'allocation mémoire sur Apple Metal (MPS) ────────────────────
# Mesuré sur M1 16 Go : les modèles résidents pèsent 6,8 Go de mémoire GPU,
# mais une seule synthèse vocale faisait monter la réservation au pilote à
# 16,7 Go — soit toute la machine. L'écart n'est pas fait de tenseurs vivants
# (6,5 Go seulement) mais de cache d'allocateur : le décodeur auto-régressif
# du TTS agrandit son cache KV d'un cran à chaque pas, donc réclame une taille
# de bloc inédite à chaque pas, et l'allocateur en conserve une par taille.
# PyTorch l'y autorise : son plafond par défaut vaut 1,7 fois la mémoire de
# travail recommandée par Metal. Le serveur finissait tué par le système.
#
# Ramené à 0,9, la réservation plafonne à 9,8 Go et reste stable d'une requête
# à l'autre, sans coût de latence mesurable (42-47 s contre 49 s auparavant sur
# une réponse type). Le ratio bas, qui déclenche la purge du cache, doit rester
# sous le ratio haut — PyTorch refuse de démarrer sinon.
#
# 0 désactiverait le plafond : à ne poser que sur une machine à grosse VRAM.
_MPS_RATIO = float(os.getenv("MPS_MEMORY_RATIO", "0.9"))
if _MPS_RATIO > 0:
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", f"{_MPS_RATIO:.2f}")
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO",
                          f"{max(0.1, _MPS_RATIO - 0.1):.2f}")


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
    # Modèle Groq : openai/gpt-oss-120b (rapide, pas de balises <think> dans la
    # réponse). Il raisonne néanmoins avant de répondre, dans un canal séparé :
    # cette trace est comptée dans LLM_MAX_TOKENS — cf. le commentaire là-bas.
    GROQ_MODEL      = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    GEMINI_MODEL    = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
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
    # Longueur visée d'un morceau de synthèse, en caractères. Le décodeur T3
    # est auto-régressif : une réponse entière tient la mémoire du GPU du début
    # à la fin, et sur une machine déjà en tension le coût s'envole. Découpée,
    # chaque passe reste bornée. 0 désactive le découpage.
    TTS_CHUNK_CHARS = int(os.getenv("TTS_CHUNK_CHARS", "200"))
    # Silence inséré entre deux morceaux au recollage, en millisecondes.
    TTS_CHUNK_PAUSE_MS = int(os.getenv("TTS_CHUNK_PAUSE_MS", "120"))

    # ── Longueur de réponse ───────────────────────────────────────────────
    # La réponse est lue à voix haute : mesuré, la synthèse coûte environ deux
    # secondes par seconde d'audio, soit ~0,14 s par caractère. Une réponse de
    # 900 caractères — il y en avait dans les 20 questions de référence — fait
    # donc attendre deux minutes devant la borne. La longueur n'est pas ici une
    # affaire de style mais de latence.
    #
    # Le plafond de jetons est un garde-fou contre une génération qui s'emballe,
    # pas l'outil de mise en forme : c'est la consigne système qui règle la
    # longueur.
    #
    # Il ne peut pas être serré. gpt-oss-120b est un modèle à raisonnement : il
    # produit une trace interne — mesuré, 750 à 840 caractères, soit environ 200
    # jetons — AVANT la réponse, et le plafond la compte. À 220 jetons, tout le
    # budget passait dans le raisonnement et la réponse revenait vide : 7 des 20
    # questions de référence sortaient sans un mot. 800 laisse la place au
    # raisonnement et à une réponse de trois phrases, et ne se déclenche que sur
    # une génération réellement anormale.
    LLM_MAX_TOKENS  = int(os.getenv("LLM_MAX_TOKENS", "800"))
    LLM_MAX_PHRASES = int(os.getenv("LLM_MAX_PHRASES", "3"))

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
