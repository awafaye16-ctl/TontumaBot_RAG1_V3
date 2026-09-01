# Prompt de réalisation — TontumaBot V3

## Contexte du projet

Construis **TontumaBot V3**, un assistant administratif sénégalais multilingue (wolof/français) avec pipeline RAG rigoureux, accessible via une API FastAPI. Le bot répond aux questions sur les démarches administratives au Sénégal, en wolof ou en français, avec une réponse vocale optionnelle.

---

## Architecture cible

```
V3/
├── app.py                          # Point d'entrée FastAPI
├── requirements.txt
├── .env / .env.example
├── download_models.py              # Script de téléchargement des modèles HF
├── wolof-whisper-small-lora/       # Poids STT locaux (M9and2M/whisper-small-wolof)
├── static/
│   ├── index.html                  # Interface chat
│   └── admin.html                  # Interface admin RAG
├── uploads/                        # Fichiers audio/docs uploadés (créé auto)
├── data/
│   ├── chroma/                     # Base vectorielle persistante ChromaDB
│   ├── seed_docs.py                # Documents de démonstration en mémoire
│   └── speaker_embedding.npy       # Embedding voix SpeechT5 (fallback)
└── src/
    ├── config.py                   # Settings chargés depuis .env
    ├── pipeline.py                 # Orchestrateur principal
    ├── ingestion.py                # Extraction texte + chunking + indexation
    ├── vectorstore.py              # ChromaDB + embeddings + BM25 + MMR
    ├── language/
    │   └── detector.py             # Détection wolof vs français
    ├── translation/
    │   └── nllb.py                 # Traduction WO↔FR via NLLB
    ├── input/
    │   └── stt.py                  # Transcription audio (Whisper)
    ├── intent/
    │   └── router.py               # Routeur d'intention (procedure | orientation)
    ├── retrieval/
    │   ├── hybrid.py               # BM25 + vectoriel
    │   ├── reranker.py             # Cross-encoder reranker
    │   └── filtered.py             # Recherche filtrée par métadonnée
    ├── generation/
    │   └── llm.py                  # Groq / Gemini / Qwen local
    ├── tts_Ooleil/
    │   └── tts.py                  # TTS : Oolel-Voices → SpeechT5 → edge-tts
    └── evaluation/
        └── ragas_eval.py           # Métriques RAGAS locales
```

---

## Pipeline de traitement complet

```
Entrée (texte ou audio)
  ↓ STT si audio          [M9and2M/whisper-small-wolof, 242M params]
Texte brut (WO ou FR)
  ↓ Détection langue       [dict de mots wolof ~120 mots spécifiques]
  Si WO → Traduction WO→FR [bilalfaye/nllb-200-distilled-600M-wo-fr-en]
  ↓ Router d'intention     [procedure | orientation]
  ↓ RAG Pipeline
      1. Hybrid search      BM25 (alpha=0.4) + vectoriel → fetch_k=20 candidats
      2. Déduplication      supprime les chunks textuellement identiques
      3. MMR (λ=0.6)        diversité anti-redondance → 8 chunks retenus
      4. Reranker           cross-encoder/ms-marco-MiniLM-L-6-v2 → top_k=3
  ↓ LLM                    Groq qwen/qwen3.6-27b | Gemini 2.0-flash | Qwen local
  Si WO → _strip_markdown() nettoie le markdown avant traduction retour
  Si WO → Traduction FR→WO [Lahad/nllb200-francais-wolof]
  ↓ TTS (optionnel)        Oolel-Voices → SpeechT5 → edge-tts (fallback cascade)
Réponse JSON
```

---

## Modèles HuggingFace utilisés

| Rôle                  | Modèle HF                                                     | Stockage                          |
| --------------------- | ------------------------------------------------------------- | --------------------------------- |
| STT wolof             | `M9and2M/whisper-small-wolof`                                 | Local `wolof-whisper-small-lora/` |
| Traduction WO→FR      | `bilalfaye/nllb-200-distilled-600M-wo-fr-en`                  | Local `src/Wo_fr_bilalfaye.../`   |
| Traduction FR→WO      | `Lahad/nllb200-francais-wolof`                                | Local `src/Fr-Wo_Lahad.../`       |
| Embeddings            | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Cache HF                          |
| Reranker              | `cross-encoder/ms-marco-MiniLM-L-6-v2`                        | Cache HF                          |
| TTS principal         | `soynade-research/Oolel-Voices`                               | Cache HF (snapshot_download)      |
| TTS fallback          | `bilalfaye/speecht5_tts-wolof-v0.2`                           | Cache HF                          |
| TTS vocoder           | `microsoft/speecht5_hifigan`                                  | Cache HF                          |
| Speaker embedding     | `Matthijs/cmu-arctic-xvectors`                                | Cache HF                          |
| LLM local (optionnel) | `Qwen/Qwen2.5-7B-Instruct`                                    | Cache HF                          |
| LLM cloud             | Groq `qwen/qwen3.6-27b`                                       | API                               |

---

## Spécifications des modules clés

### `src/config.py`
- Charge `.env` avec `os.environ.setdefault` (n'écrase pas les variables déjà définies)
- Désactive TensorFlow avant tout import (`USE_TF=0`, `USE_TORCH=1`)
- Résolution intelligente du STT : vérifie la présence de `config.json` **ET** d'un fichier de poids (`model.safetensors` ou `pytorch_model.bin`) avant d'accepter un chemin local, sinon bascule vers le Hub HF
- Propriété calculée `llm_ready` selon le provider configuré

### `src/pipeline.py`
- Fonction `_strip_markdown(text)` : retire les balises markdown (`**`, `#`, `*`) avant la traduction FR→WO car NLLB ne les gère pas
- `plain=True` passé au LLM quand `lang == "wo"` pour obtenir de la prose simple sans markdown
- Trace complète de chaque étape (latences, modèles, scores, contexte)
- Fallback seed en mémoire si ChromaDB est vide

### `src/language/detector.py`
- Dictionnaire de ~120 mots wolof très spécifiques (pronoms, verbes courants, particules, mots administratifs)
- Seuil : 1 seul mot du dictionnaire dans la question suffit pour détecter le wolof

### `src/generation/llm.py`
- Deux prompts système :
  - `SYSTEM_PROMPT` : markdown autorisé, utilisé pour les réponses en français
  - `SYSTEM_PROMPT_PLAIN` : prose simple sans markdown, utilisé quand la réponse sera traduite en wolof
- `_strip_think()` : supprime les balises `<think>...</think>` (Qwen, DeepSeek)
- Fallback : retourne le premier chunk du contexte si aucune clé API disponible
- Providers supportés : `groq` | `gemini` | `local` | fallback

### `src/vectorstore.py`
- Embeddings cosinus normalisés via `sentence-transformers`
- ID de chunk = hash MD5(texte + metadata) → idempotent, pas de doublons
- **Déduplication par texte exact** avant le MMR (corrige le bug de chunks identiques retournés)
- `hybrid_search` : `alpha=0.4` (légère préférence sémantique sur lexical)
- `mmr_search` : `λ=0.6` (pertinence > diversité)
- Recherche filtrée par métadonnée pour l'intention "orientation"

### `src/ingestion.py`
- Chunking sémantique sur frontières de phrase (`CHUNK_SIZE=512`, `CHUNK_OVERLAP=80`, `MIN_CHUNK_LEN=40`)
- Supporte PDF (pypdf), TXT, MD
- Métadonnées enrichies par chunk : `document_id`, `title`, `source`, `chunk_index`, `total_chunks`, `nb_chars`, `nb_words`, `added` (ISO 8601 UTC)

### `src/translation/nllb.py`
- Chargement lazy (au premier appel) des deux modèles NLLB, restent en mémoire
- Codes de langue : `wol_Latn` (wolof) et `fra_Latn` (français)
- `num_beams=4`, `max_new_tokens=128`, `early_stopping=True`
- CPU par défaut, GPU si disponible

### `src/input/stt.py`
- Chargement lazy de Whisper (au premier appel audio)
- Chargement depuis dossier local `STT_MODEL_PATH` en priorité, sinon Hub HF
- Resample audio à 16 kHz via `librosa` (supporte WAV, MP3, M4A, WebM)
- Langue forcée optionnelle via `forced_decoder_ids`

### `src/tts_Ooleil/tts.py`
- Cascade de fallback : `Oolel-Voices` → `SpeechT5` → `edge-tts (fr-FR-DeniseNeural)`
- `TTS_ENGINE` configurable dans `.env` : `oolel` | `speecht5` | `edge`
- Oolel-Voices chargé via `snapshot_download` depuis HF Hub, cherche le voice prompt `8_1_c.wav`
- SpeechT5 : découpe le texte en segments ≤ 80 chars sur frontières de phrase, normalise le volume (pic cible 0.9, gain max 6×)

### `src/evaluation/ragas_eval.py`
Métriques calculées **localement** (sans LLM externe) :
- `context_precision` : proportion de chunks pertinents dans le top-K (F1 mots > 0.1 vs référence)
- `context_recall` : proportion des mots de la référence couverts par les chunks
- `faithfulness` : proportion de phrases de la réponse supportées par le contexte (F1 > 0.15)
- `answer_relevance` : recouvrement de mots-clés entre question et réponse
- `answer_correctness` : F1 mots entre réponse générée et réponse de référence
- `abstention_rate` : taux de refus corrects sur les cas sans information

---

## API FastAPI — Endpoints

| Méthode | Endpoint                 | Description                                                    |
| ------- | ------------------------ | -------------------------------------------------------------- |
| GET     | `/`                      | Interface chat (`index.html`)                                  |
| GET     | `/admin`                 | Interface admin RAG (`admin.html`)                             |
| GET     | `/health`                | Statut de tous les modèles + stats ChromaDB                    |
| POST    | `/ask`                   | Question texte JSON `{question, provider?, tts?, tts_engine?}` |
| POST    | `/ask/audio`             | Audio multipart → STT → RAG → réponse                          |
| GET     | `/translate`             | Test traduction NLLB `?text=...&direction=wo2fr\|fr2wo`        |
| GET     | `/response.wav`          | Dernier audio TTS généré                                       |
| POST    | `/admin/documents`       | Ingère un fichier (TXT/PDF/DOCX) ou texte brut                   |
| GET     | `/admin/documents`       | Liste des documents indexés                                    |
| DELETE  | `/admin/documents/{id}`  | Supprime un document                                           |
| POST    | `/admin/documents/clear` | Vide toute la base vectorielle                                 |
| POST    | `/eval/ragas`            | Évaluation RAGAS sur jeu de test JSON                          |

---

## Format de réponse `/ask`

```json
{
  "response_fr": "Réponse en français",
  "response_wo": "Réponse en wolof (si question WO)",
  "response":    "Réponse dans la langue de l'utilisateur",
  "audio":       "/static/response.wav (si tts=true)",
  "trace": {
    "input_lang": "wo",
    "wolof_to_french": {
      "model": "bilalfaye/nllb-200-distilled-600M-wo-fr-en",
      "result": "Je veux un certificat de naissance.",
      "latency_ms": 5170
    },
    "intent": "procedure",
    "retrieval": {
      "n_candidates": 20,
      "n_mmr": 8,
      "n_reranked": 3,
      "reranker_scores": [4.42, 2.27, -0.65],
      "chunks": [{"rank": 1, "score": 4.42, "text": "..."}],
      "source": "chromadb"
    },
    "context": "texte des top-3 chunks envoyés au LLM",
    "llm": {"provider": "groq", "latency_ms": 3748},
    "french_to_wolof": {
      "model": "Lahad/nllb200-francais-wolof",
      "latency_ms": 4431
    },
    "tts": {"engine": "oolel-voices", "latency_ms": 0},
    "total_latency_ms": 19861
  }
}
```

---

## Fichier `.env`

```env
# ── LLM ──────────────────────────────────────────────
LLM_PROVIDER=groq
GROQ_API_KEY=<ta_clé_groq>
GEMINI_API_KEY=<ta_clé_gemini>
LOCAL_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
LOCAL_LLM_QUANT=4bit

# ── Traduction ────────────────────────────────────────
NLLB_WO_FR_MODEL=bilalfaye/nllb-200-distilled-600M-wo-fr-en
NLLB_FR_WO_MODEL=Lahad/nllb200-francais-wolof

# ── STT ───────────────────────────────────────────────
STT_MODEL_PATH=./wolof-whisper-small-lora
STT_LANGUAGE=

# ── Embeddings ────────────────────────────────────────
EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# ── TTS ───────────────────────────────────────────────
TTS_ENGINE=oolel
OOLEL_TTS_REPO=soynade-research/Oolel-Voices
WOLOF_TTS_MODEL=bilalfaye/speecht5_tts-wolof-v0.2

# ── Reranker ──────────────────────────────────────────
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANKER_TOP_K=3

# ── Serveur ───────────────────────────────────────────
HOST=0.0.0.0
PORT=8008
```

---

## `requirements.txt`

```
torch
transformers==4.46.3
sentencepiece
accelerate
bitsandbytes
safetensors
librosa>=0.10.2
rank-bm25
sentence-transformers
chromadb
pypdf
fastapi
uvicorn
python-multipart
groq
google-generativeai
huggingface_hub
soundfile
edge-tts
diffusers==0.29.0
conformer==0.3.2
omegaconf
scipy
numpy<2.0
datasets
langdetect
pillow
```

---

## Installation et lancement

```bash
# 1. Cloner le repo
git clone https://github.com/awafaye16-ctl/TontumaBot_RAG1.git
cd TontumaBot_RAG1/V3

# 2. Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer l'environnement
cp .env.example .env
# Renseigner GROQ_API_KEY et/ou GEMINI_API_KEY dans .env

# 5. Télécharger les modèles locaux (NLLB WO↔FR + STT Whisper)
python download_models.py

# 6. Lancer le serveur
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Accès :
- **Chat** → http://localhost:8008
- **Admin RAG** → http://localhost:8008/admin
- **Health** → http://localhost:8008/health
- **API docs** → http://localhost:8008/docs

---

## Points de vigilance et corrections apportées

| Problème                                                     | Correction                                                                                                                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Chunks dupliqués dans le MMR                                 | Déduplication par texte exact avant l'algorithme MMR dans `vectorstore.py`                                                                              |
| LLM répond `INFORMATION_ABSENTE` sur contexte partiel        | Prompt moins strict : synthétise à partir d'infos partiellement liées                                                                                   |
| Réponse markdown cassée par NLLB lors de la traduction FR→WO | `_strip_markdown()` dans `pipeline.py` + `SYSTEM_PROMPT_PLAIN` pour les questions wolof                                                                 |
| Détecteur de langue trop limité (50 mots)                    | Dictionnaire élargi à ~120 mots wolof spécifiques                                                                                                       |
| Poids STT absents du dossier local (Git LFS non initialisé)  | Vérification de la présence des poids (`model.safetensors`) en plus de `config.json` dans `config.py` ; téléchargement automatique depuis Hub si absent |
| Port déjà utilisé au redémarrage                             | L'app tourne avec `--reload` via uvicorn, ne pas relancer `app.py` manuellement                                                                         |
| `esc()` crash sur valeur `undefined` côté frontend           | `esc = s => (s == null ? '' : String(s)).replace(...)`                                                                                                  |
