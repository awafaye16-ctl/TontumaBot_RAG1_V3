#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  TontumaBot V3 — démarrage en natif, sans Docker
#
#  Sur une machine neuve :
#      ./demarrer.sh
#
#  Options :
#      --reinstall     recréer l'environnement virtuel de zéro
#      --skip-models   ne pas télécharger les poids (s'ils sont déjà en cache)
#      --no-start      tout préparer, sans lancer le serveur
#      -h | --help
#
#  Pourquoi en natif plutôt qu'en conteneur : un conteneur n'a pas accès au GPU
#  d'un Mac. Mesuré sur la même machine, la synthèse vocale passe d'un facteur
#  2,0 à 6,8 fois la durée de l'audio produit quand elle tombe sur le CPU — une
#  réponse parlée de quinze secondes demande alors 100 s au lieu de 30 s.
#  Pour une borne en service sur un Mac, c'est ce script qu'il faut utiliser ;
#  setup.sh et docker-compose.yml servent à livrer sur un serveur Linux.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

cd "$(dirname "$0")"

REINSTALL=0
SKIP_MODELS=0
NO_START=0

while [ $# -gt 0 ]; do
    case "$1" in
        --reinstall)   REINSTALL=1 ;;
        --skip-models) SKIP_MODELS=1 ;;
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

VENV=venv

# ── 1. Interpréteur Python ───────────────────────────────────────────────────
#  3.10 au minimum : le code annote les types avec la syntaxe `str | None`
#  (src/device.py, src/tts_Ooleil/tts.py…), évaluée à l'exécution et rejetée
#  par 3.9. On préfère la version la plus récente disponible.
info "Recherche d'un interpréteur Python (3.10 ou plus)"

PY=""
for candidat in python3.13 python3.12 python3.11 python3.10 python3; do
    command -v "$candidat" >/dev/null 2>&1 || continue
    if "$candidat" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 10) else 1)' 2>/dev/null; then
        PY="$candidat"
        break
    fi
done

[ -n "$PY" ] || die "Aucun Python 3.10+ trouvé. Installez-le (macOS : 'brew install python@3.12')."
ok "Python : $("$PY" --version 2>&1) ($(command -v "$PY"))"

# ── 2. Espace disque ─────────────────────────────────────────────────────────
#  ~2 Go de dépendances (torch en pèse la moitié) et ~4 Go de poids de modèles.
AVAIL_GB=$(( $(df -Pk . | awk 'NR==2 {print $4}') / 1024 / 1024 ))
if [ "$AVAIL_GB" -lt 10 ]; then
    warn "Seulement ${AVAIL_GB} Go libres — comptez ~2 Go de dépendances et ~4 Go de modèles."
fi

# ── 3. Environnement virtuel ─────────────────────────────────────────────────
if [ "$REINSTALL" -eq 1 ] && [ -d "$VENV" ]; then
    info "Suppression de l'environnement existant (--reinstall)"
    rm -rf "$VENV"
fi

if [ ! -d "$VENV" ]; then
    info "Création de l'environnement virtuel"
    "$PY" -m venv "$VENV"
    ok "Environnement créé dans ./$VENV"
fi

PIP="$VENV/bin/pip"
PYV="$VENV/bin/python"
[ -x "$PYV" ] || die "L'environnement ./$VENV est incomplet. Relancez avec --reinstall."

# ── 4. Dépendances ───────────────────────────────────────────────────────────
#  Réinstaller à chaque démarrage coûte une minute pour rien. On note l'empreinte
#  de requirements.txt après une installation réussie : tant qu'elle n'a pas
#  changé et que les paquets répondent, on passe directement à la suite.
EMPREINTE_FICHIER="$VENV/.dependances-installees"
EMPREINTE=$(cksum requirements.txt | awk '{print $1}')
DEPS_OK=0

#  Deux façons de conclure que rien n'est à installer :
#
#  - l'empreinte enregistrée correspond et les paquets répondent : cas normal
#    d'un démarrage quotidien ;
#  - aucune empreinte n'a été enregistrée, mais les paquets répondent quand
#    même : c'est un environnement monté à la main avant l'existence de ce
#    script. Le réinstaller risquerait de faire monter torch en version et de
#    casser une installation qui marche. On l'adopte en notant son empreinte.
#
#  Dans les deux cas, une modification de requirements.txt relance l'installation.
if "$PYV" -c 'import torch, fastapi, chromadb, soundfile' >/dev/null 2>&1; then
    if [ ! -f "$EMPREINTE_FICHIER" ]; then
        echo "$EMPREINTE" > "$EMPREINTE_FICHIER"
        DEPS_OK=1
        info "Environnement existant adopté (aucune réinstallation)"
    elif [ "$(cat "$EMPREINTE_FICHIER")" = "$EMPREINTE" ]; then
        DEPS_OK=1
    fi
fi

if [ "$DEPS_OK" -eq 1 ]; then
    ok "Dépendances déjà installées"
else
    #  Le choix de la roue torch dépend de la machine :
    #
    #  - macOS : les roues PyPI embarquent Metal, donc l'accélération GPU. C'est
    #    tout l'intérêt d'un démarrage natif, il ne faut surtout pas les
    #    remplacer par une roue « CPU ».
    #  - Linux avec carte NVIDIA : les roues PyPI par défaut embarquent CUDA.
    #  - Linux x86 sans GPU : l'index CPU de PyTorch évite de télécharger
    #    2,5 Go de bibliothèques CUDA inutiles.
    #  - Linux ARM sans GPU : l'index CPU n'y publie torchaudio que jusqu'à la
    #    version 2.0.2, incompatible avec un torch récent — on reste sur PyPI,
    #    dont les roues ARM sont de toute façon sans CUDA.
    OS=$(uname -s)
    ARCH=$(uname -m)
    TORCH_INDEX=""
    if [ "$OS" = "Darwin" ]; then
        info "Installation de PyTorch (roues macOS, accélération Metal)"
    elif command -v nvidia-smi >/dev/null 2>&1; then
        info "Installation de PyTorch (roues CUDA, GPU NVIDIA détecté)"
    elif [ "$ARCH" = "x86_64" ]; then
        TORCH_INDEX="--index-url https://download.pytorch.org/whl/cpu"
        info "Installation de PyTorch (roues CPU, ~200 Mo au lieu de 2,5 Go)"
    else
        info "Installation de PyTorch (roues PyPI, architecture $ARCH)"
    fi

    "$PIP" install --upgrade pip >/dev/null
    # shellcheck disable=SC2086
    "$PIP" install $TORCH_INDEX torch torchaudio

    #  bitsandbytes ne sert qu'au LLM local quantifié (LLM_PROVIDER=local) et
    #  n'existe que pour CUDA : l'écarter d'office évite un échec d'installation
    #  qui bloquerait tout le reste sur un Mac.
    info "Installation des dépendances du projet"
    REQ_TMP=$(mktemp)
    grep -v '^bitsandbytes' requirements.txt > "$REQ_TMP"
    "$PIP" install -r "$REQ_TMP"
    rm -f "$REQ_TMP"

    if command -v nvidia-smi >/dev/null 2>&1; then
        "$PIP" install bitsandbytes || warn "bitsandbytes indisponible — sans effet sauf si LLM_PROVIDER=local"
    fi

    echo "$EMPREINTE" > "$EMPREINTE_FICHIER"
    ok "Dépendances installées"
fi

# ── 5. Configuration ─────────────────────────────────────────────────────────
if [ ! -f .env ]; then
    [ -f .env.example ] || die "Ni .env ni .env.example — impossible de configurer le projet."
    cp .env.example .env
    warn "Fichier .env créé depuis .env.example — renseignez-y votre clé API."
fi

#  `|| true` sur chaque lecture : sans correspondance, grep sort en erreur et
#  `set -e` interromprait le script alors qu'une clé absente est justement le
#  cas qu'on veut signaler.
lire_env() { grep "^$1=" .env 2>/dev/null | head -1 | cut -d= -f2- | tr -d ' \r' || true; }

PROVIDER=$(lire_env LLM_PROVIDER); PROVIDER=${PROVIDER:-groq}
case "$PROVIDER" in
    groq)   [ -n "$(lire_env GROQ_API_KEY)" ]   || warn "GROQ_API_KEY est vide dans .env — le LLM ne répondra pas." ;;
    gemini) [ -n "$(lire_env GEMINI_API_KEY)" ] || warn "GEMINI_API_KEY est vide dans .env — le LLM ne répondra pas." ;;
    local)  info "LLM local : les poids seront téléchargés au premier appel." ;;
esac

mkdir -p data/chroma uploads certs static
ok "Configuration prête (LLM : $PROVIDER)"

# ── 6. Backend de calcul ─────────────────────────────────────────────────────
#  Annoncé avant tout le reste : un repli silencieux sur CPU multiplie les temps
#  de synthèse par trois, et c'est la première chose à vérifier quand la borne
#  paraît lente.
info "Backend de calcul détecté"
(cd src && "../$VENV/bin/python" device.py) || warn "Détection du backend impossible — le serveur choisira au démarrage."

# ── 7. Poids des modèles ─────────────────────────────────────────────────────
if [ "$SKIP_MODELS" -eq 1 ]; then
    warn "Téléchargement des modèles ignoré (--skip-models)"
elif [ -f src/stt_wolof-hubert-ctc/config.json ]; then
    ok "Modèles déjà présents"
else
    info "Téléchargement des modèles (~4 Go, une seule fois)"
    "$PYV" download_models.py
    ok "Modèles téléchargés"
fi

# ── 8. Corpus indexé ─────────────────────────────────────────────────────────
#  data/chroma/ n'est pas suivi par git : un dépôt fraîchement cloné démarre
#  donc avec une base vide, sans la moindre erreur — la borne répond alors
#  « je n'ai pas trouvé cette information » à toutes les questions. Le silence
#  serait le pire des diagnostics : on compte les fragments et on le dit.
CHUNKS=$("$PYV" - <<'PY' 2>/dev/null || echo 0
import sqlite3, pathlib
base = pathlib.Path("data/chroma/chroma.sqlite3")
if not base.exists():
    print(0)
else:
    with sqlite3.connect(f"file:{base}?mode=ro", uri=True) as c:
        try:
            print(c.execute("select count(*) from embeddings").fetchone()[0])
        except sqlite3.Error:
            print(0)
PY
)

if [ "${CHUNKS:-0}" -gt 0 ]; then
    ok "Corpus indexé : ${CHUNKS} fragments"
else
    warn "Base documentaire vide : la borne démarrera mais ne saura répondre à rien."
    echo "     Copiez data/chroma/ depuis une machine où le corpus est indexé,"
    echo "     ou déposez les documents via /admin puis relancez l'ingestion."
fi

# ── 9. Démarrage ─────────────────────────────────────────────────────────────
PORT=$(lire_env PORT); PORT=${PORT:-8008}

if [ "$NO_START" -eq 1 ]; then
    info "Préparation terminée. Démarrage : $VENV/bin/python app.py"
    exit 0
fi

echo
info "Démarrage du serveur sur le port ${PORT}"
echo "     Le préchargement des modèles prend environ une minute sur GPU,"
echo "     plusieurs minutes sur CPU. Le serveur répond dès qu'il affiche"
echo "     « Application startup complete »."
echo
echo "     Borne  : http://localhost:${PORT}/borne"
echo "     Démo   : http://localhost:${PORT}/"
echo "     Santé  : http://localhost:${PORT}/health"
echo
echo "     Arrêt  : Ctrl+C"
echo

#  exec : le serveur remplace le script, donc Ctrl+C l'atteint directement au
#  lieu d'être intercepté par le shell qui l'aurait lancé.
exec "$PYV" app.py
