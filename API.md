# TontumaBot V2 — Documentation API

> Assistant administratif sénégalais multilingue (Wolof / Français) avec RAG, Reranker et TTS dual engine.

**Base URL** : `http://localhost:8008`
**Version** : 2.0.0

---

## Table des matières

1. [Pipeline](#pipeline)
2. [Authentification](#authentification)
3. [Endpoints](#endpoints)
   - [GET /health](#get-health)
   - [POST /ask](#post-ask)
   - [POST /ask/audio](#post-askaudio)
   - [GET /translate](#get-translate)
   - [GET /response.wav](#get-responsewav)
   - [POST /admin/documents](#post-admindocuments)
   - [GET /admin/documents](#get-admindocuments)
   - [DELETE /admin/documents/{document_id}](#delete-admindocumentsdocument_id)
   - [POST /admin/documents/clear](#post-admindocumentsclear)
4. [Pipeline Trace](#pipeline-trace)
5. [Exemples](#exemples)

---

## Pipeline

```
Entrée (texte ou audio)
  │
  ├─ Texte brut
  │   └─▶ Détection langue (wolof / français)
  │
  ├─ Audio
  │   └─▶ STT (Whisper wolof) ──▶ texte
  │
  ▼
Traduction (si wolof → français)     [NLLB 600M]
  │
  ▼
Détection d'intention                 [procedure / orientation]
  │
  ▼
Retriever (hybride BM25 + vectoriel)  [ChromaDB + embeddings]
  │
  ▼
Reranker (cross-encoder)              [ms-marco-MiniLM-L-6-v2]
  │
  ▼
Génération LLM                       [Groq / Gemini]
  │
  ├─ Si entrée wolof : Traduction FR → WO
  │
  ▼
TTS optionnel                        [Oolel-Voices / SpeechT5]
  │
  ▼
Réponse (texte + audio)
```

---

## Authentification

Aucune. Tous les endpoints sont publics (CORS `*`).

---

## Endpoints

### `GET /health`

Statut du service et de ses composants.

**Réponse** `200 OK`

```json
{
  "status": "ok",
  "version": "2.0.0",
  "llm_provider": "groq",
  "llm_ready": true,
  "tts_engine": "oolel",
  "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
  "n_documents": 5,
  "n_chunks": 12
}
```

| Champ | Type | Description |
|---|---|---|
| `status` | string | `ok` si le serveur tourne |
| `version` | string | Version de l'API |
| `llm_provider` | string | `groq` ou `gemini` |
| `llm_ready` | bool | `true` si la clé API est configurée |
| `tts_engine` | string | `oolel`, `speecht5` ou `edge-tts` |
| `reranker` | string | Modèle cross-encoder utilisé |
| `n_documents` | int | Nombre de documents indexés |
| `n_chunks` | int | Nombre total de chunks vectorisés |

---

### `POST /ask`

Pose une question en texte (wolof ou français). Le pipeline détecte la langue, traduit si besoin, recherche le contexte, génère la réponse, et optionnellement produit l'audio.

**Content-Type** : `application/json`

**Body**

```json
{
  "question": "Comment créer une entreprise au Sénégal ?",
  "provider": "groq",
  "tts": true,
  "tts_engine": "oolel"
}
```

| Champ | Type | Requis | Défaut | Description |
|---|---|---|---|---|
| `question` | string | **oui** | — | Question en wolof ou français |
| `provider` | string | non | depuis `.env` | `groq` ou `gemini` |
| `tts` | bool | non | `false` | Générer l'audio en réponse |
| `tts_engine` | string | non | depuis `.env` | `oolel` ou `speecht5` |

**Réponse** `200 OK`

```json
{
  "trace": {
    "input_lang": "wo",
    "wolof_to_french": {
      "query": "Comment créer une entreprise au Sénégal ?",
      "seconds": 1.23
    },
    "intent": "procedure",
    "search": "hybride (seed)",
    "reranker": {
      "input_docs": 4,
      "output_docs": 3,
      "scores": [7.06, -0.91, -10.75]
    },
    "context": "Créer une entreprise au Sénégal nécessite...",
    "french_to_wolof": {
      "seconds": 0.98
    },
    "tts": "oolel-voices"
  },
  "response_fr": "Pour créer une entreprise au Sénégal...",
  "response": "Ngir sos boppam entreprise ci Senegal...",
  "response_wo": "Ngir sos boppam entreprise ci Senegal...",
  "audio": "/Users/.../static/response.wav"
}
```

| Champ | Type | Description |
|---|---|---|
| `trace` | object | Détails du pipeline (voir [Pipeline Trace](#pipeline-trace)) |
| `response_fr` | string | Réponse en français |
| `response` | string | Réponse dans la langue de l'utilisateur (WO ou FR) |
| `response_wo` | string | Réponse en wolof (uniquement si `input_lang == "wo"`) |
| `audio` | string | Chemin du fichier audio généré (uniquement si `tts: true`) |

**Erreurs**

| Code | Cause |
|---|---|
| `400` | Question vide |
| `500` | Erreur interne (LLM, TTS...) |

---

### `POST /ask/audio`

Envoie un fichier audio pour transcription (STT) puis traitement pipeline complet.

**Content-Type** : `multipart/form-data`

**Body**

| Champ | Type | Requis | Description |
|---|---|---|---|
| `file` | file | **oui** | Fichier audio (wav, mp3, webm, m4a...) |
| `tts` | bool | non | Générer l'audio en réponse (défaut: `false`) |
| `tts_engine` | string | non | `oolel` ou `speecht5` |

**Réponse** : Identique à `POST /ask`.

**Erreurs**

| Code | Cause |
|---|---|
| `500` | Échec de la transcription STT |

---

### `GET /translate`

Endpoint de test pour la traduction NLLB (sans LLM).

**Paramètres query** :

| Param | Type | Défaut | Description |
|---|---|---|---|
| `text` | string | `"Jërejëf lool ci dimbal bi."` | Texte à traduire |
| `direction` | string | `wo2fr` | `wo2fr` (wolof→français) ou `fr2wo` (français→wolof) |

**Réponse** `200 OK`

```json
{
  "input": "Jërejëf lool ci dimbal bi.",
  "output": "Merci beaucoup pour l'aide.",
  "direction": "wo->fr",
  "seconds": 0.45
}
```

---

### `GET /response.wav`

Retourne le dernier fichier audio généré par le TTS.

**Réponse** : Fichier audio WAV (`audio/wav`)

**Erreurs**

| Code | Cause |
|---|---|
| `404` | Aucun audio généré |

---

### `POST /admin/documents`

Ajoute un document dans la base vectorielle (ChromaDB). Le document est découpé en chunks et vectorisé.

**Content-Type** : `multipart/form-data`

**Body** (l'une ou l'autre)

| Champ | Type | Requis | Description |
|---|---|---|---|
| `file` | file | **oui*** | Fichier `.txt`, `.md` ou `.pdf` |
| `text` | string | **oui*** | Texte brut à indexer |
| `title` | string | non | Titre du document |

*\*Fournir `file` OU `text`, pas les deux.*

**Réponse** `200 OK`

```json
{
  "ok": true,
  "chunks": 3,
  "title": "Guide administratif"
}
```

**Erreurs**

| Code | Cause |
|---|---|
| `400` | Aucun fichier/texte, format non supporté, ou texte vide |
| `500` | Erreur d'extraction (PDF corrompu, etc.) |

---

### `GET /admin/documents`

Liste tous les documents indexés dans ChromaDB.

**Réponse** `200 OK`

```json
{
  "documents": [
    {
      "id": "a1b2c3d4e5f6",
      "title": "Guide administratif",
      "chunks": 3,
      "added": "/home/user"
    }
  ],
  "total_chunks": 3
}
```

---

### `DELETE /admin/documents/{document_id}`

Supprime un document et tous ses chunks de la base vectorielle.

**Réponse** `200 OK`

```json
{
  "ok": true,
  "deleted_chunks": 3
}
```

**Erreurs**

| Code | Cause |
|---|---|
| `404` | Document introuvable |

---

### `POST /admin/documents/clear`

Supprime **tous** les documents de la base vectorielle.

**Réponse** `200 OK`

```json
{
  "ok": true,
  "deleted_chunks": 12
}
```

---

## Pipeline Trace

L'objet `trace` dans la réponse de `/ask` et `/ask/audio` contient les étapes exécutées :

| Clé | Type | Description |
|---|---|---|
| `input_lang` | string | Langue détectée : `wo` ou `fr` |
| `wolof_to_french` | object | `{query, seconds}` — traduction WO→FR (si input wolof) |
| `intent` | string | `procedure` ou `orientation` |
| `search` | string | Type de recherche utilisée |
| `reranker` | object | `{input_docs, output_docs, scores}` — résultats du reranker |
| `context` | string | Contexte envoyé au LLM |
| `french_to_wolof` | object | `{seconds}` — traduction FR→WO (si input wolof) |
| `tts` | string | Source audio : `oolel-voices`, `speecht5-wolof` ou `edge-tts-fr` |

---

## Exemples

### Question texte (français)

```bash
curl -X POST http://localhost:8008/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Combien coûte un passeport ?"}'
```

### Question texte (wolof) avec audio

```bash
curl -X POST http://localhost:8008/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Dama beug wout kayitu juddu ?",
    "tts": true,
    "tts_engine": "oolel"
  }'
```

### Question audio (micro)

```bash
curl -X POST http://localhost:8008/ask/audio \
  -F "file=@recording.wav" \
  -F "tts=true" \
  -F "tts_engine=oolel"
```

### Ajouter un document

```bash
curl -X POST http://localhost:8008/admin/documents \
  -F "file=@guide.pdf" \
  -F "title=Guide administratif"
```

### Lister les documents

```bash
curl http://localhost:8008/admin/documents
```

### Traduction rapide

```bash
curl "http://localhost:8008/translate?text=Jërejëf+lool+ci+dimbal+bi.&direction=wo2fr"
```

---

## Configuration

Les paramètres sont définis dans le fichier `.env` à la racine du projet V2 :

| Variable | Valeur par défaut | Description |
|---|---|---|
| `GROQ_API_KEY` | — | Clé API Groq |
| `GEMINI_API_KEY` | — | Clé API Google Gemini |
| `LLM_PROVIDER` | `groq` | Fournisseur LLM |
| `TTS_ENGINE` | `oolel` | Moteur TTS : `oolel` ou `speecht5` |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Modèle reranker |
| `RERANKER_TOP_K` | `3` | Nombre de documents après reranking |
| `HOST` | `0.0.0.0` | Adresse de binding |
| `PORT` | `8008` | Port du serveur |

---

## Dépendances

```
fastapi, uvicorn, python-multipart
chromadb, sentence-transformers, rank-bm25, pypdf
openai-whisper, SpeechRecognition
transformers==4.46.3, torch, sentencepiece
groq, google-generativeai
edge-tts, soundfile, librosa>=0.10.2
diffusers==0.29.0, conformer==0.3.2, s3tokenizer
langdetect
```

> **Python 3.11+** requis. `transformers 4.46.3` segfault sur Python 3.13 avec certains modèles.
