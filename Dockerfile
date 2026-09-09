# ─────────────────────────────────────────────────────────────────────────────
#  TontumaBot V3 — image de l'API FastAPI (CPU)
#
#  Les poids des modèles ne sont PAS embarqués dans l'image : ils sont
#  téléchargés dans des volumes au premier lancement (cf. download_models.py
#  et le service `download-models` du docker-compose.yml).
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── Dépendances système ──────────────────────────────────────────────────────
#  ffmpeg     : décodage des audios envoyés par la borne (webm/opus, m4a…)
#               via librosa/audioread
#  libsndfile1: lecture/écriture WAV par soundfile
#  git        : certains paquets HuggingFace le réclament au chargement
#  build-essential : compilation des rares paquets sans wheel
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        libgomp1 \
        git \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Environnement Python ─────────────────────────────────────────────────────
#  HOME=/app : src/tts_Ooleil/tts.py cherche le snapshot Oolel-Voices dans
#              ~/.cache/huggingface/hub — HOME et HF_HOME pointent donc sur le
#              même volume.
#  USE_TF=0   : TensorFlow/Keras jamais chargé (cf. app.py), ~20 s au démarrage
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/app \
    HF_HOME=/app/.cache/huggingface \
    USE_TF=0 \
    USE_TORCH=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# ── Dépendances Python ───────────────────────────────────────────────────────
COPY requirements.txt .

# torch/torchaudio depuis l'index CPU : ~200 Mo au lieu de ~2,5 Go avec CUDA.
RUN pip install --upgrade pip \
 && pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio

# bitsandbytes ne sert qu'au LLM local quantifié (LLM_PROVIDER=local) et n'a
# pas de wheel sur toutes les architectures. Installé à part, en best-effort,
# pour qu'une absence de wheel n'échoue pas la construction.
RUN grep -v '^bitsandbytes' requirements.txt > /tmp/req.txt \
 && pip install -r /tmp/req.txt \
 && (pip install bitsandbytes || echo "[build] bitsandbytes ignoré (indisponible sur cette architecture)")

# ── Code applicatif ──────────────────────────────────────────────────────────
COPY . .

# Dossiers écrits à l'exécution (montés en volumes par docker-compose)
RUN mkdir -p /app/uploads /app/data/chroma /app/resultats_ragas /app/.cache/huggingface \
 && useradd --create-home --uid 1000 tontuma \
 && chown -R tontuma:tontuma /app
USER tontuma

EXPOSE 8008

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -fsS http://localhost:${PORT:-8008}/health || exit 1

# app.py lit .env, applique HOST/PORT et le HTTPS optionnel (SSL_CERTFILE)
CMD ["python", "app.py"]
