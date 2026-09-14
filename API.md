# TontumaBot V3 — Documentation API

> Assistant administratif multilingue (Wolof / Français) : RAG hybride, STT par
> langue, synthèse vocale diffusée par morceaux.

**Base URL** : `http://localhost:8008` · **Version** : `3.0.0`

Ce document est le contrat pour qui **consomme** l'API. Le fonctionnement
interne — latences mesurées, plafond mémoire, journalisation — est décrit dans
le `README.md`.

---

## Ce qui change depuis la V2

| V2 | V3 |
|---|---|
| `POST /ask` renvoyait un JSON | **renvoie un flux SSE** ; le JSON arrive dans l'événement `result` |
| `tts_engine` : `oolel` \| `speecht5` | SpeechT5 retiré ; Oolel-Voices seul, plus de paramètre |
| `GET /response.wav` | fichiers servis sur `/static/{nom}.wav` |
| trace en `seconds` | trace en `latency_ms` |
| — | synthèse **diffusée par morceaux** (`tts_chunk`) |
| — | mémoire conversationnelle (`session_id`, endpoints `/session/*`) |
| — | `lang` sur `/ask/audio` : choisit le moteur STT |

---

## Table des matières

1. [Authentification](#authentification)
2. [Le protocole SSE](#le-protocole-sse)
3. [Événements](#événements)
4. [Endpoints](#endpoints)
5. [La trace](#la-trace)
6. [Clients d'exemple](#clients-dexemple)
7. [Codes d'erreur](#codes-derreur)

---

## Authentification

**Aucune.** Tous les endpoints sont publics et CORS est ouvert à `*`.

> ⚠️ Cela inclut `/admin/documents` : n'importe qui pouvant joindre le service
> peut **ajouter, lister ou effacer** la base documentaire. Le fichier `api.key`
> présent dans le dépôt n'est lu par aucune route. En exposition réseau,
> protégez `/admin/*` en amont (reverse proxy, filtrage IP) ou n'exposez que
> `localhost`.

---

## Le protocole SSE

`POST /ask` et `POST /ask/audio` répondent en `text/event-stream`. L'envoi reste
un POST classique ; c'est la **réponse** qui est un flux.

```
event: <nom>\n
data: <json>\n
\n
```

Un flux se termine toujours par `done`, y compris après une erreur.

**Pourquoi un flux** : la chaîne complète dure de 6 à 120 secondes selon la
longueur de la réponse et la synthèse vocale. Attendre le résultat final
laisserait l'usager devant un écran muet pendant une minute ; les événements
intermédiaires permettent d'afficher l'avancement, puis de commencer à parler
avant la fin de la synthèse.

Un client qui ignore les événements intermédiaires et n'écoute que `result`
reste parfaitement valide.

---

## Événements

### `status` — progression

```
event: status
data: {"step": "retrieval"}
```

Séquence : `start` → (`stt`) → `detect` → (`alerte_langue`) → (`translate_in`) →
`intent` → `retrieval` → `llm` → (`translate_out`) → (`alerte_nombres`) →
(`tts`) → (`tts_chunk` × N) → puis `result` → `done`.

Les étapes entre parenthèses n'apparaissent que si elles ont lieu. **Traitez
toute étape inconnue comme ignorable** : de nouvelles peuvent apparaître.

| `step` | Charge utile | Sens |
|---|---|---|
| `start` | `question`, `provider`, `tts` | requête acceptée ; sur `/ask/audio`, `question` porte la transcription |
| `stt` | `lang` | transcription en cours, dans la langue déclarée |
| `detect` | `lang`, `source` | `source` vaut `indice` (langue déclarée) ou `détection` (déduite du texte) |
| `translate_in` | — | wolof → français |
| `intent` | `intent` | `procedure` ou `orientation` |
| `retrieval` | — | recherche documentaire |
| `llm` | `provider` | rédaction |
| `translate_out` | — | français → wolof |
| `tts` | `response`, `response_fr`, `response_wo`, `lang` | **le texte de la réponse, disponible avant l'audio** |
| `tts_chunk` | `index`, `total`, `audio_url`, `latency_ms` | un morceau audio prêt à lire |
| `alerte_langue` | `declaree`, `detectee`, `scores` | la transcription contredit la langue annoncée |
| `alerte_nombres` | `controles`, `suspects`, `ajoutes`, `coherent` | un nombre du français ne se retrouve pas en wolof |

### L'étape `tts` porte le texte

```
event: status
data: {"step": "tts", "response": "Ngir mu mën a dem lopitaal bi...",
       "response_fr": "Pour se rendre à l'hôpital...", "response_wo": "...",
       "lang": "wo"}
```

Le `result` n'arrive qu'une fois l'audio produit — jusqu'à deux minutes plus
tard sur une réponse longue. **Affichez le texte dès cet événement**, sinon
votre interface parlera avant de montrer ce qu'elle dit.

### `tts_chunk` — la synthèse arrive par morceaux

```
event: status
data: {"step": "tts_chunk", "index": 1, "total": 5,
       "audio_url": "/static/response_3b5f134a_01.wav", "latency_ms": 37340.2}
```

Chaque morceau est lisible immédiatement, **dans l'ordre d'`index`**. Un client
qui les enchaîne commence à parler dès le premier ; mesuré sur une réponse en 5
morceaux, le premier son est disponible à 37 s contre 153 s pour l'énoncé
complet.

Trois points à connaître :

- Une réponse courte tient en **un seul morceau** : `index` et `total` valent 1,
  et le fichier du morceau **est** le fichier final — aucun doublon n'est écrit.
- `latency_ms` se compte depuis le début de la synthèse, pas de la requête.
- La synthèse produit environ deux fois moins vite qu'on n'écoute : votre file
  de lecture se videra en cours de route. Restez en attente du morceau suivant
  plutôt que de rendre la main.

Un client qui ignore ces événements attend simplement `audio_url` dans
`result`, qui reste la réponse entière en un seul fichier.

### `alerte_langue`

```
event: status
data: {"step": "alerte_langue", "declaree": "wo", "detectee": "fr",
       "scores": {"wolof": 0, "francais": 16, "n_mots": 9}}
```

Émis quand la transcription contredit nettement la langue déclarée — typiquement
un usager qui s'est trompé de bouton. **Rien n'est corrigé** : le mal est fait à
la transcription, et la langue déclarée continue de gouverner tout le pipeline.
L'événement existe pour que vous puissiez l'expliquer à l'usager, dont la
réponse va paraître absurde.

### `alerte_nombres`

NLLB réécrit parfois les montants et se trompe : `1000 francs CFA` est ressorti
en `junniy dërëm`, soit 5000 F. Quand la valeur d'origine a disparu, rien ne
peut la réparer en aval — le pipeline signale au lieu de corriger.

### `result` — le résultat complet

```
event: result
data: {
  "response":     "Ngir mu mën a dem...",          // dans la langue de l'usager
  "response_fr":  "Pour se rendre à l'hôpital...", // pivot, toujours présent
  "response_wo":  "Ngir mu mën a dem...",          // absent si la question était en français
  "qr_code":      "iVBORw0KGgo...",                // PNG base64, procédures seulement
  "audio_url":    "/static/response_ab12cd34.wav", // null si tts=false
  "lang":         "wo",
  "memory":       {"count": 4, "max": 10, "reset": false},
  "trace":        { ... }
}
```

`memory.reset` à `true` signifie que l'historique **vient d'être vidé** : la
question suivante ne bénéficiera plus du contexte. Prévenez l'usager, sinon ses
questions de suite cesseront d'être comprises sans explication.

### `error` puis `done`

```
event: error
data: {"message": "STT échoué : ..."}

event: done
data: {}
```

---

## Endpoints

### `GET /health`

État du service, sans effet de bord.

```json
{
  "status": "ok", "version": "3.0.0",
  "llm_provider": "groq", "llm_ready": true,
  "tts": "oolel-voices",
  "nllb_model": "bilalfaye/nllb-200-distilled-600M-wo-fr-en",
  "stt_wo": "./src/stt_wolof-hubert-ctc", "stt_fr": "openai/whisper-small",
  "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
  "device": {"torch": "2.11.0", "cuda": false, "mps": true, "auto": "mps",
             "par_composant": {"nllb": "mps", "tts": "mps"}},
  "n_documents": 2, "n_chunks": 32,
  "memory": {"enabled": true, "max_messages": 10, "sessions": 3}
}
```

`llm_ready` à `false` signifie qu'aucune clé d'API n'est configurée : le service
répond quand même, en renvoyant le passage documentaire le plus pertinent au
lieu d'une réponse rédigée.

`device.auto` est la première chose à regarder si le service paraît lent : un
repli silencieux sur `cpu` triple les temps.

---

### `POST /ask` — question écrite

`Content-Type: application/json` → **réponse SSE**.

| Champ | Type | Requis | Défaut | Description |
|---|---|---|---|---|
| `question` | string | **oui** | — | en wolof ou en français |
| `provider` | string | non | `.env` | `groq` \| `gemini` \| `local` |
| `tts` | bool | non | `false` | produire aussi l'audio |
| `lang` | string | non | `null` | `wo` \| `fr` — **sinon détection automatique sur le texte** |
| `session_id` | string | non | `null` | mémoire conversationnelle |

Sur ce chemin, omettre `lang` est le comportement normal : la détection
travaille sur du texte, où elle est fiable.

```bash
curl -N -X POST http://localhost:8008/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Comment obtenir une copie de mon dossier médical ?","tts":false}'
```

---

### `POST /ask/audio` — question vocale

`multipart/form-data` → **réponse SSE**.

| Champ | Type | Requis | Défaut | Description |
|---|---|---|---|---|
| `file` | fichier | **oui** | — | WAV, MP3, M4A, WebM |
| `lang` | string | non | `wo` | **choisit le moteur STT** — voir ci-dessous |
| `tts` | bool | non | `false` | produire aussi l'audio |
| `provider` | string | non | `.env` | `groq` \| `gemini` \| `local` |
| `session_id` | string | non | `null` | mémoire conversationnelle |

> **`lang` n'est pas optionnel en pratique.** Il sélectionne le moteur de
> transcription — `wo` → Wolof-HuBERT-CTC, `fr` → Whisper — et ce choix se fait
> *avant* qu'il existe une transcription à analyser : la langue d'un fichier
> audio n'est pas détectable ici. Le défaut `wo` signifie qu'une question posée
> en français **sans** ce champ sera transcrite par le modèle wolof, puis
> répondue en wolof. Faites déclarer la langue par l'usager (deux boutons, un
> sélecteur) et transmettez-la.

```bash
curl -N -X POST http://localhost:8008/ask/audio \
  -F "file=@question.webm" -F "lang=fr" -F "tts=true"
```

---

### Mémoire conversationnelle

Passez le même `session_id` — un UUID que vous générez — d'une question à
l'autre. Le serveur retient l'historique en **français**, langue pivot du
pipeline, et s'en sert pour comprendre les questions de suite (« et combien ça
coûte ? »). L'historique n'est jamais une source de faits : seule la base
documentaire fonde une réponse.

Au-delà de `MEMORY_MAX_MESSAGES` (10 par défaut, soit 5 échanges), la mémoire
est **vidée** — la session continue, aucune erreur n'est levée. C'est
`memory.reset` dans `result` qui vous en informe.

| Endpoint | Effet |
|---|---|
| `GET /session/{id}` | `{session_id, messages: [{role, content}], count, max, resets, total}` — jamais 404 : une session inconnue renvoie un état vide |
| `POST /session/{id}/reset` | vide l'historique, garde la session → `{ok, existed}` |
| `DELETE /session/{id}` | oublie la session → `{ok, existed}` |

La mémoire vit en RAM, dans le processus : un redémarrage l'efface, et plusieurs
instances ne la partagent pas.

---

### `GET /translate` — traduction seule

Test rapide de NLLB, sans RAG ni LLM.

| Paramètre | Défaut | Valeurs |
|---|---|---|
| `text` | `Jërejëf lool ci dimbal bi.` | texte à traduire |
| `direction` | `wo2fr` | `wo2fr` \| `fr2wo` |

```json
{"input": "...", "output": "...", "direction": "wo→fr",
 "model": "bilalfaye/nllb-200-distilled-600M-wo-fr-en", "seconds": 1.2}
```

---

### Fichiers audio

Servis en statique sur `/static/{nom}.wav`, tels que fournis par `audio_url` et
`tts_chunk.audio_url`. **Ne construisez pas ces URL vous-même** : les noms
portent un identifiant aléatoire.

Ces fichiers ne sont jamais purgés automatiquement : une réponse en produit N+1
(les morceaux, plus l'assemblage). Prévoyez le ménage côté exploitation.

---

### Administration RAG

Aucune authentification — voir l'avertissement en tête de document.

| Endpoint | Corps | Réponse |
|---|---|---|
| `POST /admin/documents` | `file` (txt/md/pdf) **ou** `text`, plus `title` optionnel | `{ok, chunks, title}` |
| `GET /admin/documents` | — | `{documents: [{id, title, source, chunks, added}], total_chunks}` |
| `DELETE /admin/documents/{document_id}` | — | `{ok, deleted_chunks}` · `404` si inconnu |
| `POST /admin/documents/clear` | — | `{ok, deleted_chunks}` — **efface tout** |

À l'ingestion, le document est découpé en fragments de 512 caractères avec 80 de
recouvrement, enrichi de métadonnées, encodé et indexé.

```bash
curl -X POST http://localhost:8008/admin/documents \
  -F "file=@procedures.pdf" -F "title=Procédures hospitalières"
```

---

### `POST /eval/ragas` — évaluation

Lance une évaluation sur un jeu de test fourni.

```json
{"provider": "groq",
 "test_cases": [{"question": "...", "reference_answer": "...",
                 "category": "procedure", "language": "fr"}]}
```

> Les métriques de ce module sont calculées par **recouvrement lexical de mots**,
> pas par un juge. Elles donnent un ordre de grandeur, pas une note. Les scripts
> de `benchmark/` produisent une évaluation plus solide.

---

### Pages servies

`/` (démo), `/borne` (borne tactile), `/borne/simple` (variante allégée),
`/admin` (gestion documentaire).

---

## La trace

Présente dans `result.trace`. Toutes les durées sont en **millisecondes**.

| Clé | Contenu |
|---|---|
| `input_lang`, `input_lang_source` | langue retenue, et son origine (`indice` \| `détection`) |
| `wolof_to_french`, `french_to_wolof` | `{model, latency_ms, result?, nombres?, realignement?}` |
| `intent`, `intent_latency_ms` | intention détectée |
| `retrieval` | `{n_candidates, n_mmr, n_reranked, latency_hybrid_ms, latency_mmr_ms, latency_rerank_ms, latency_total_ms, reranker_scores, chunks}` |
| `retrieval.chunks[]` | `{rank, id, score, text}` — **`id` identifie le fragment**, ce qui permet de savoir après coup quel passage a nourri la réponse |
| `llm` | `{provider, latency_ms, history_messages}` |
| `tts` | `{engine, latency_ms, n_morceaux, texte_synthetise, ...}` |
| `alerte_langue` | présent seulement en cas de contradiction |
| `total_latency_ms` | bout en bout |

Ordre de grandeur mesuré sur une réponse type, matériel Apple M1 : traduction
FR→WO ~3 300 ms, LLM ~1 200 ms, traduction WO→FR ~1 000 ms, recherche ~90 ms,
intention ~50 ms. La synthèse vocale, quand elle est demandée, dépasse tout le
reste : comptez environ deux secondes par seconde d'audio produit.

---

## Clients d'exemple

### JavaScript (navigateur)

```js
const res = await fetch('/ask', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ question, tts: true, session_id: sessionId }),
});

const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
let tampon = '';
for (;;) {
  const { value, done } = await reader.read();
  if (done) break;
  tampon += value;
  const blocs = tampon.split('\n\n');
  tampon = blocs.pop();                    // fragment incomplet conservé
  for (const bloc of blocs) {
    const ev = (bloc.match(/^event: (.*)$/m) || [])[1];
    const data = JSON.parse((bloc.match(/^data: (.*)$/m) || [])[1] || '{}');
    if (ev === 'status' && data.step === 'tts')       afficher(data.response);
    if (ev === 'status' && data.step === 'tts_chunk') enfiler(data.audio_url);
    if (ev === 'result')                              finaliser(data);
  }
}
```

Le découpage sur `\n\n` avec conservation du fragment incomplet n'est pas
facultatif : un événement peut arriver coupé en deux lectures.

### Java (Spring WebClient)

```java
webClient.post().uri("/ask")
    .contentType(MediaType.APPLICATION_JSON)
    .bodyValue(Map.of("question", question, "tts", true, "session_id", sessionId))
    .retrieve()
    .bodyToFlux(ServerSentEvent.class)
    .doOnNext(ev -> {
        if ("result".equals(ev.event())) traiterResultat(ev.data());
    })
    .blockLast();
```

### Ligne de commande

```bash
curl -N -X POST http://localhost:8008/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Test","tts":false}'
```

`-N` désactive la mise en tampon de curl, sans quoi le flux n'apparaît qu'à la fin.

---

## Codes d'erreur

| Code | Quand |
|---|---|
| `200` | y compris pour un flux qui se terminera par `event: error` |
| `400` | question vide, format de fichier refusé, texte vide à l'ingestion |
| `404` | document inconnu à la suppression |
| `500` | erreur interne d'ingestion |

**Une erreur de pipeline n'est pas un code HTTP.** Le flux ayant déjà commencé
avec un `200`, l'échec arrive dans `event: error` puis `event: done`. Traitez
les deux.
