#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  TontumaBot V3 — installation Docker de bout en bout
#
#  Sur une machine neuve :
#      ./setup.sh
#
#  Options :
#      --skip-models   ne pas télécharger les poids (s'ils sont déjà en cache)
#      --rebuild       reconstruire l'image sans cache
#      --no-start      s'arrêter après le build et le téléchargement
#      -h | --help
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

cd "$(dirname "$0")"

SKIP_MODELS=0
REBUILD=0
NO_START=0

while [ $# -gt 0 ]; do
    case "$1" in
        --skip-models) SKIP_MODELS=1 ;;
        --rebuild)     REBUILD=1 ;;
        --no-start)    NO_START=1 ;;
        -h|--help)     sed -n '3,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Option inconnue : $1 (voir --help)" >&2; exit 1 ;;
    esac
    shift
done

info() { printf '\033[1;34m▸\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m⚠\033[0m  %s\n' "$*"; }
die()  { printf '\033[1;31m✗\033[0m %s\n' "$*" >&2; exit 1; }
ok()   { printf '\033[1;32m✓\033[0m %s\n' "$*"; }

# ── 1. Prérequis ─────────────────────────────────────────────────────────────
info "Vérification des prérequis"

command -v docker >/dev/null 2>&1 || die "Docker n'est pas installé — https://docs.docker.com/get-docker/"

if docker compose version >/dev/null 2>&1; then
    DC="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
    DC="docker-compose"
else
    die "Le plugin Docker Compose est absent — https://docs.docker.com/compose/install/"
fi

docker info >/dev/null 2>&1 || die "Le daemon Docker ne répond pas. Démarrez Docker Desktop (ou 'sudo systemctl start docker') puis relancez."

ok "Docker prêt ($DC)"

# ── 2. Espace disque ─────────────────────────────────────────────────────────
# L'image pèse ~1 Go et les poids ~4 Go : il faut de la marge.
AVAIL_KB=$(df -Pk . | awk 'NR==2 {print $4}')
AVAIL_GB=$(( AVAIL_KB / 1024 / 1024 ))
if [ "$AVAIL_GB" -lt 8 ]; then
    warn "Seulement ${AVAIL_GB} Go libres. Comptez ~1 Go pour l'image et ~4 Go pour les modèles."
    warn "Libérez de la place, ou récupérez le cache de build Docker : docker builder prune -f"
    printf "Continuer quand même ? [y/N] "
    read -r reply
    case "$reply" in [yYoO]*) ;; *) die "Interrompu." ;; esac
else
    ok "Espace disque : ${AVAIL_GB} Go libres"
fi

# ── 3. Configuration (.env) ──────────────────────────────────────────────────
info "Configuration"

if [ ! -f .env ]; then
    [ -f .env.example ] || die ".env.example introuvable — dépôt incomplet ?"
    cp .env.example .env
    ok ".env créé depuis .env.example"
fi

# Cache HuggingFace partagé avec l'hôte : les modèles déjà présents sur la
# machine ne sont pas retéléchargés, et ceux que l'on télécharge servent aussi
# à une exécution native.
HF_DEFAULT="${HF_HOME:-$HOME/.cache/huggingface}"
if grep -q '^HF_CACHE_DIR=.\+' .env; then
    HF_CACHE_DIR=$(grep '^HF_CACHE_DIR=' .env | head -1 | cut -d= -f2-)
else
    HF_CACHE_DIR="$HF_DEFAULT"
    if grep -q '^HF_CACHE_DIR=' .env; then
        # Ligne présente mais vide : on la remplit (sed -i diffère BSD/GNU)
        tmp=$(mktemp)
        sed "s|^HF_CACHE_DIR=.*|HF_CACHE_DIR=${HF_CACHE_DIR}|" .env > "$tmp" && mv "$tmp" .env
    else
        printf '\n# Cache HuggingFace partagé avec l'"'"'hôte\nHF_CACHE_DIR=%s\n' "$HF_CACHE_DIR" >> .env
    fi
fi
export HF_CACHE_DIR
mkdir -p "$HF_CACHE_DIR"
ok "Cache HuggingFace : $HF_CACHE_DIR"

# Clés API : sans clé le LLM ne répondra pas (sauf LLM_PROVIDER=local).
PROVIDER=$(grep '^LLM_PROVIDER=' .env | head -1 | cut -d= -f2- | tr -d ' ')
GROQ=$(grep  '^GROQ_API_KEY='   .env | head -1 | cut -d= -f2- | tr -d ' ')
GEMINI=$(grep '^GEMINI_API_KEY=' .env | head -1 | cut -d= -f2- | tr -d ' ')
case "${PROVIDER:-groq}" in
    groq)   [ -n "$GROQ" ]   || warn "GROQ_API_KEY est vide dans .env — le LLM ne répondra pas." ;;
    gemini) [ -n "$GEMINI" ] || warn "GEMINI_API_KEY est vide dans .env — le LLM ne répondra pas." ;;
    local)  info "LLM local : les poids Qwen seront téléchargés au premier appel." ;;
esac

# ── 4. Dossiers montés ───────────────────────────────────────────────────────
# Docker crée les bind mounts manquants en root : on les crée nous-mêmes pour
# qu'ils appartiennent à l'utilisateur courant.
mkdir -p data/chroma uploads certs src/stt_wolof-hubert-ctc
ok "Dossiers de travail prêts"

# ── 5. Construction de l'image ───────────────────────────────────────────────
info "Construction de l'image (première fois : ~5 min)"
if [ "$REBUILD" -eq 1 ]; then
    $DC build --no-cache api
else
    $DC build api
fi
ok "Image construite"

# ── 6. Téléchargement des modèles ────────────────────────────────────────────
# ~4 Go : STT wolof (HuBERT-CTC), Whisper, NLLB WO↔FR. Les embeddings, le
# reranker et le TTS Oolel-Voices arrivent au premier démarrage via le cache.
if [ "$SKIP_MODELS" -eq 1 ]; then
    warn "Téléchargement des modèles ignoré (--skip-models)"
elif [ -f src/stt_wolof-hubert-ctc/config.json ]; then
    ok "Modèles déjà présents — téléchargement ignoré"
else
    info "Téléchargement des modèles (~4 Go, une seule fois)"
    $DC run --rm download-models
    ok "Modèles téléchargés"
fi

# ── 7. Démarrage ─────────────────────────────────────────────────────────────
if [ "$NO_START" -eq 1 ]; then
    info "Build terminé. Démarrage : $DC up -d"
    exit 0
fi

info "Démarrage du conteneur"
$DC up -d api

PORT=$(grep '^PORT=' .env | head -1 | cut -d= -f2- | tr -d ' ')
PORT=${PORT:-8008}

# Le warm-up précharge tous les modèles : plusieurs minutes sur CPU.
info "Warm-up en cours (préchargement des modèles, jusqu'à 10 min sur CPU)…"
for _ in $(seq 1 120); do
    if curl -fsS --max-time 5 "http://localhost:${PORT}/health" >/dev/null 2>&1; then
        echo
        ok "TontumaBot V3 est prêt"
        echo
        echo "   Borne    : http://localhost:${PORT}/borne"
        echo "   Admin    : http://localhost:${PORT}/admin"
        echo "   Santé    : http://localhost:${PORT}/health"
        echo
        echo "   Logs     : $DC logs -f api"
        echo "   Arrêt    : $DC down"
        exit 0
    fi
    # Un conteneur qui meurt ne guérit pas : inutile d'attendre la fin du délai.
    if [ "$(docker inspect -f '{{.State.Running}}' tontumabot-v3 2>/dev/null)" != "true" ]; then
        echo
        die "Le conteneur s'est arrêté. Journal : $DC logs --tail 50 api"
    fi
    printf '.'
    sleep 5
done

echo
die "Toujours pas de réponse sur /health après 10 min. Journal : $DC logs --tail 50 api"
