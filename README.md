# TontumaBot V3

Assistant administratif multilingue (Wolof / Français) avec RAG, STT, TTS et API REST + SSE.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TontumaBot V3 (FastAPI)                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │   STT    │→ │  Detect  │→ │Translate │→ │  Intent  │→ │   RAG    │     │
│  │ (Whisper)│  │  Langue  │  │ WO → FR  │  │  Router  │  │ (Hybrid  │     │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │  + MMR   │     │
│                                                           │  + Rerank)    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  └──────────┘     │
│  │   TTS    │← │Translate │← │   LLM    │← │ Context  │                    │
│  │Oolel-Voices│  │ FR → WO │  │ (Groq/   │  │ + Prompt │                    │
│  └──────────┘  └──────────┘  │  Gemini/ │  └──────────┘                    │
│                              │  Local)  │                                  │
│                              └──────────┘                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Stack technique

| Composant | Technologie |
|-----------|-------------|
| API       | FastAPI + Uvicorn (REST pour l'envoi, SSE pour la réponse) |
| LLM       | Groq (openai/gpt-oss-120b), Gemini, Local (Qwen2.5-7B 4bit) |
| Embeddings| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 |
| Vector DB | ChromaDB (persistant) |
| Retrieval | Hybrid BM25 + Vectoriel → MMR → Cross-encoder reranker (index + embeddings cachés) |
| Traduction| NLLB-200-distilled-600M (WO↔FR) |
| STT       | Whisper-small-wolof (M9and2M / LoRA local) |
| TTS       | **Oolel-Voices** (voice cloning, soynade-research/Oolel-Voices) |
| Frontend  | HTML/JS vanilla (REST + SSE) |

## Démarrage rapide

```bash
# 1. Dépendances
pip install -r requirements.txt

# 2. Configuration (.env existe déjà, adaptez si besoin)
cp .env.example .env
# éditez .env : GROQ_API_KEY, GEMINI_API_KEY, etc.

# 3. Lancement
python app.py
# → http://localhost:8008
```

> **Warm-up au démarrage** : les modèles (embedder, reranker, NLLB, et
> optionnellement STT/TTS) sont préchargés au lancement (~20-30 s) pour éviter
> la latence de chargement sur la première requête. Désactivable via
> `WARMUP_ON_START=false` (voir `.env`).

## Endpoints REST

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET`   | `/` | Interface web |
| `GET`   | `/health` | État du service |
| `POST`  | `/ask` | Question texte → réponse **SSE** (voir plus bas) |
| `POST`  | `/ask/audio` | Audio → STT → Pipeline → réponse **SSE** |
| `GET`   | `/static/{fichier}.wav` | Récupérer un audio TTS généré (via `audio_url`) |
| `POST`  | `/admin/documents` | Ingérer document (TXT/MD/PDF) |
| `GET`   | `/admin/documents` | Lister documents indexés |
| `DELETE`| `/admin/documents/{id}` | Supprimer un document |
| `POST`  | `/translate` | Test traduction NLLB |
| `POST`  | `/eval/ragas` | Évaluation RAGAS |

> `/ask` et `/ask/audio` répondent en **SSE** (`text/event-stream`), pas en JSON
> d'un bloc. Voir la section **API SSE** ci-dessous pour le format des événements.

---

## API SSE (streaming de la réponse)

Le modèle est **REST pour l'envoi** (`POST`) et **SSE (Server-Sent Events) pour la réponse**.
`/ask` et `/ask/audio` renvoient un flux `text/event-stream` : la progression du
pipeline est émise étape par étape, suivie du résultat final. Le WebSocket a été
retiré au profit de ce modèle, plus simple à opérer et à faire passer les proxies.

> L'audio TTS n'est **pas** envoyé dans le flux (SSE = texte). Le résultat contient
> un champ `audio_url` pointant vers un fichier WAV servi sur `/static`, que le
> client récupère par un simple `GET`.

### Requête

```
POST /ask
Content-Type: application/json

{ "question": "Comment faire un passeport ?", "tts": true, "provider": "groq" }
```

| Champ | Type | Requis | Défaut | Description |
|-------|------|--------|--------|-------------|
| `question` | string | ✅ | — | Question en wolof ou français |
| `tts` | boolean | ❌ | `false` | Générer l'audio TTS |
| `provider` | string | ❌ | config | `groq` \| `gemini` \| `local` |

### Flux d'événements

Chaque événement suit le format SSE `event: <nom>\ndata: <json>\n\n`.

**Statuts de progression** (`event: status`) :
```
event: status
data: {"step": "start", "question": "...", "tts": true, "provider": "groq"}

event: status
data: {"step": "detect", "lang": "fr"}
```

Étapes possibles : `start` → (`stt`) → `detect` → (`translate_in`) → `intent` →
`retrieval` → `llm` → (`translate_out`) → (`tts`) → puis `result` → `done`.
Les étapes entre parenthèses n'apparaissent que si pertinentes (audio, entrée wolof, TTS).

**Résultat final** (`event: result`) :
```
event: result
data: {
  "response": "Pour obtenir un passeport...",
  "response_fr": "Pour obtenir un passeport...",
  "response_wo": "Baati passeport bi...",
  "qr_code": "iVBORw0KGgo...",        // base64 PNG (procédures)
  "audio_url": "/static/response_ab12cd34.wav",  // null si tts=false
  "lang": "fr",
  "trace": { "total_latency_ms": 1200, ... }
}
```

**Erreur** (`event: error`) puis `event: done` :
```
event: error
data: {"message": "..."}

event: done
data: {}
```

### Test en ligne de commande

```bash
curl -N -X POST http://localhost:8008/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Comment faire un passeport ?", "tts": false, "provider": "groq"}'
```

### Client Java (Spring Boot / WebClient)

`EventSource` natif ne fait que du `GET` ; pour un `POST` + SSE, on lit le flux
directement (`WebClient` en Spring, ou `fetch` + `ReadableStream` côté navigateur).

```java
WebClient client = WebClient.create("http://localhost:8008");
client.post().uri("/ask")
    .contentType(MediaType.APPLICATION_JSON)
    .bodyValue(Map.of("question", "Où déposer ma demande de CNI ?", "tts", true, "provider", "groq"))
    .retrieve()
    .bodyToFlux(ServerSentEvent.class)
    .doOnNext(evt -> {
        switch (evt.event()) {
            case "status" -> log.info("étape: {}", evt.data());
            case "result" -> {
                JsonNode r = objectMapper.readTree((String) evt.data());
                String reponse  = r.get("response").asText();
                String audioUrl = r.get("audio_url").isNull() ? null : r.get("audio_url").asText();
                // audioUrl → GET http://localhost:8008{audioUrl} pour récupérer le WAV
            }
            case "error" -> log.error("erreur: {}", evt.data());
        }
    })
    .blockLast();
```

### Client JavaScript (navigateur, fetch + stream)

```js
const resp = await fetch('/ask', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ question: 'Comment créer une entreprise ?', tts: true, provider: 'groq' }),
});

const reader = resp.body.pipeThrough(new TextDecoderStream()).getReader();
let buffer = '';
for (;;) {
  const { value, done } = await reader.read();
  if (done) break;
  buffer += value;
  const events = buffer.split('\n\n');
  buffer = events.pop();               // dernier fragment incomplet
  for (const block of events) {
    const ev  = block.match(/^event: (.*)$/m)?.[1];
    const dat = JSON.parse(block.match(/^data: (.*)$/m)?.[1] || '{}');
    if (ev === 'status')  console.log('étape', dat.step);
    if (ev === 'result') {
      console.log('réponse', dat.response);
      if (dat.audio_url) new Audio(dat.audio_url).play();   // GET du WAV
    }
  }
}
```

---

## Variables d'environnement (.env)

```ini
# LLM
LLM_PROVIDER=groq
GROQ_API_KEY=...
GEMINI_API_KEY=...
GROQ_MODEL=openai/gpt-oss-120b      # modèle Groq par défaut (non-reasoning, rapide)
GEMINI_MODEL=gemini-2.0-flash
LOCAL_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct

# Traduction
NLLB_WO_FR_MODEL=bilalfaye/nllb-200-distilled-600M-wo-fr-en
NLLB_NUM_BEAMS=2       # 2 = qualité, 1 = greedy (plus rapide)

# STT
STT_MODEL_PATH=./src/stt_wolof-whisper-small-lora

# Embeddings
EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# TTS (Oolel-Voices)
OOLEL_TTS_REPO=soynade-research/Oolel-Voices
TTS_SPEED=1.0          # 1.0 = normal, >1 ralentit
TTS_N_STEPS=10         # 10 = qualité max, 4 = rapide (~2.5x)

# Reranker
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANKER_TOP_K=3

# Warm-up des modèles au démarrage (évite la latence de chargement au 1er appel)
WARMUP_ON_START=true
WARMUP_STT=true        # false pour ne pas précharger le STT (chargé au 1er audio)
WARMUP_TTS=true        # false pour ne pas précharger le TTS (chargé au 1er TTS)

# Serveur
HOST=0.0.0.0
PORT=8008
```

---

## Modèles utilisés

| Tâche | Modèle | Source |
|-------|--------|--------|
| LLM (cloud) | openai/gpt-oss-120b | Groq |
| LLM (cloud) | gemini-2.0-flash | Google |
| LLM (local) | Qwen2.5-7B-Instruct (4bit) | HuggingFace |
| Embeddings | paraphrase-multilingual-MiniLM-L12-v2 | SBERT |
| Traduction WO↔FR | NLLB-200-distilled-600M | bilalfaye (HF) |
| STT Wolof | Whisper-small-wolof | M9and2M / LoRA local |
| TTS (principal) | **Oolel-Voices** | soynade-research (HF) |
| Reranker | ms-marco-MiniLM-L-6-v2 | cross-encoder |

---

## Structure du projet

```
TontumaBot_RAG1_V3/
├── app.py                     # FastAPI (REST + SSE) + warm-up des modèles
├── download_models.py         # Pré-téléchargement des modèles HF
├── pipeline_rag.py            # Script pipeline autonome (hors serveur)
├── requirements.txt
├── .env                       # Variables d'environnement
├── static/
│   ├── index.html             # Interface web (REST + SSE)
│   └── admin.html             # Interface d'administration RAG
├── src/
│   ├── config.py              # Configuration centralisée (.env)
│   ├── pipeline.py            # Pipeline RAG complet (+ callback SSE)
│   ├── vectorstore.py         # ChromaDB + BM25/embeddings cachés + MMR
│   ├── ingestion.py           # Indexation documents (TXT/MD/PDF)
│   ├── input/stt.py           # STT Whisper wolof
│   ├── language/detector.py   # Détection de langue
│   ├── translation/nllb.py    # Traduction WO↔FR (NLLB)
│   ├── intent/router.py       # Router d'intention
│   ├── retrieval/
│   │   ├── hybrid.py          # BM25 + vectoriel
│   │   ├── filtered.py        # Recherche filtrée (orientation)
│   │   └── reranker.py        # Cross-encoder
│   ├── generation/llm.py      # LLM (Groq / Gemini / local)
│   ├── tts_Ooleil/tts.py      # TTS Oolel-Voices
│   └── evaluation/ragas_eval.py  # Évaluation RAGAS
└── data/                      # Documents seed + ChromaDB persistant
```

---

## Développement

```bash
# Lancer en mode dev (reload auto)
uvicorn app:app --reload --host 0.0.0.0 --port 8008

# Test SSE en ligne de commande (-N = pas de buffering)
curl -N -X POST http://localhost:8008/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Test","tts":false,"provider":"groq"}'
```

---

## Licence

Projet interne — usage administratif sénégalais.