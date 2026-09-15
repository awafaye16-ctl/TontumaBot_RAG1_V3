# TontumaBot V3 — Documentation API

> Assistant administratif multilingue (Wolof / Français) : RAG hybride, STT par
> langue, synthèse vocale diffusée par morceaux.

**Base URL** : `http://localhost:8008` · **Version** : `3.1.0`

Ce document est le contrat pour qui **consomme** l'API. Le fonctionnement
interne — latences mesurées, plafond mémoire, journalisation — est décrit dans
le `README.md`.

---

## Ce qui change depuis la V3.0

| V3.0 | V3.1 |
|---|---|
| base documentaire unique | **une base vectorielle par organisation**, isolée sur le disque |
| — | `organization_id` (UUID) **requis** sur `/ask` et `/ask/audio` |
| `/admin/documents*` | `/admin/organizations/{organization_id}/documents*` |
| ingestion par fichier ou texte | **+ ingestion par pointeur** MinIO (`bucket` + `objectKey`) |
| `{ok, chunks, title}` | `{ok, document_id, organization_id, chunks, replaced, …}` |
| identifiant de document calculé en interne | **fourni par l'appelant** — republier remplace au lieu d'empiler |
| traçabilité enfouie dans `trace.retrieval.chunks` | **champ `sources`** au premier niveau de `result` |
| `EMBED_MODEL` modifié → collection purgée | → `409`, **aucune donnée touchée** |
| `n_documents` / `n_chunks` globaux sur `/health` | `organizations: {total, chargees}` + `storage` |

Le contrat d'intégration destiné au backend est détaillé dans
**`INTEGRATION_BACKEND.md`**.

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

> ⚠️ Cela inclut les routes d'administration : n'importe qui pouvant joindre le
> service peut **ajouter, lister ou effacer** la base documentaire de n'importe
> quelle organisation. Le fichier `api.key` présent dans le dépôt n'est lu par
> aucune route. En exposition réseau, protégez `/admin/*` en amont (reverse
> proxy, filtrage IP) ou n'exposez que `localhost`.
>
> **Conséquence pour le multi-tenant** : l'`organization_id` est une clé de
> routage, pas une frontière de sécurité. L'isolation des bases empêche une
> réponse de citer le document d'une autre structure ; elle n'empêche pas un
> appelant de se présenter comme la structure de son choix. La frontière, c'est
> le réseau.

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
  "sources":      [ ... ],                         // traçabilité RAG, voir plus bas
  "memory":       {"count": 4, "max": 10, "reset": false},
  "trace":        { ... }
}
```

### `result.sources` — quels documents ont nourri la réponse

```json
"sources": [
  {"document_id": "d4c1f0a2-8e35-4b77-9a10-5f3c2e8b1d69",
   "title": "Demande de copie de dossier médical", "category": "procedure",
   "chunk_id": "doc-3b5f134a2c8e9d01", "rank": 1, "score": 7.42}
]
```

`document_id` est **l'identifiant fourni à l'ingestion** : c'est la clé de
jointure avec la base de l'appelant. `chunk_id` identifie un fragment de 512
caractères et change à chaque réindexation — il sert au débogage, pas à la
jointure.

`score` est la sortie brute du cross-encoder : **non bornée, souvent négative,
non comparable d'une question à l'autre**. Elle ordonne, elle ne note pas. Pour
afficher une pertinence à un usager, utilisez `rank`.

La liste est **vide** quand l'organisation n'a aucun document ou qu'aucun
passage n'est pertinent — cas normal, fréquent sur une structure qui vient
d'être créée. Un même `document_id` peut y figurer deux fois si deux de ses
fragments ont été retenus.

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
  "organizations": {"total": 12, "chargees": 3},
  "storage": {"configure": true, "joignable": true},
  "memory": {"enabled": true, "max_messages": 10, "sessions": 3}
}
```

`llm_ready` à `false` signifie qu'aucune clé d'API n'est configurée : le service
répond quand même, en renvoyant le passage documentaire le plus pertinent au
lieu d'une réponse rédigée.

`device.auto` est la première chose à regarder si le service paraît lent : un
repli silencieux sur `cpu` triple les temps.

`organizations.total` compte les bases présentes sur le disque, `chargees`
celles actuellement en mémoire (cache LRU, plafonné par
`MAX_ORGANISATIONS_EN_CACHE`). Les comptages de documents sont désormais propres
à chaque organisation : voir `GET /admin/organizations`.

`storage` décrit l'accès au stockage objet. `{"configure": false}` signifie que
l'ingestion par pointeur est indisponible ; l'ingestion directe, elle, continue
de fonctionner.

---

### `POST /ask` — question écrite

`Content-Type: application/json` → **réponse SSE**.

| Champ | Type | Requis | Défaut | Description |
|---|---|---|---|---|
| `question` | string | **oui** | — | en wolof ou en français |
| `organization_id` | UUID | **oui** | — | base de connaissances interrogée — `400` si absent ou mal formé |
| `provider` | string | non | `.env` | `groq` \| `gemini` \| `local` |
| `tts` | bool | non | `false` | produire aussi l'audio |
| `lang` | string | non | `null` | `wo` \| `fr` — **sinon détection automatique sur le texte** |
| `session_id` | string | non | `null` | mémoire conversationnelle |

Sur ce chemin, omettre `lang` est le comportement normal : la détection
travaille sur du texte, où elle est fiable.

```bash
curl -N -X POST http://localhost:8008/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Comment obtenir une copie de mon dossier médical ?",
       "organization_id":"9b7d1e44-2c08-4a17-b3f5-6e2d9c447a10","tts":false}'
```

---

### `POST /ask/audio` — question vocale

`multipart/form-data` → **réponse SSE**.

| Champ | Type | Requis | Défaut | Description |
|---|---|---|---|---|
| `file` | fichier | **oui** | — | WAV, MP3, M4A, WebM |
| `organization_id` | UUID | **oui** | — | base de connaissances interrogée |
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
  -F "file=@question.webm" -F "lang=fr" -F "tts=true" \
  -F "organization_id=9b7d1e44-2c08-4a17-b3f5-6e2d9c447a10"
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

Toutes ces routes sont **scopées par organisation**, et l'`organization_id` est
dans le **chemin**. Ce n'est pas cosmétique : un segment oublié donne un `404`,
là où un champ de corps oublié donnerait une opération silencieusement globale.

| Endpoint | Corps | Réponse |
|---|---|---|
| `POST /admin/organizations/{org}/documents/from-storage` | JSON `{documentId, bucket, objectKey, title?, category?}` | `{ok, document_id, organization_id, title, category, chunks, replaced, added}` |
| `POST /admin/organizations/{org}/documents` | `file` (txt/md/pdf) **ou** `text`, plus `title`, `document_id`, `category` optionnels | idem |
| `GET /admin/organizations/{org}/documents` | — | `{organization_id, documents: [{id, title, source, category, chunks, added}], total_documents, total_chunks}` |
| `DELETE /admin/organizations/{org}/documents/{document_id}` | — | `{ok, organization_id, deleted_chunks}` · `404` si inconnu |
| `POST /admin/organizations/{org}/documents/clear` | — | `{ok, organization_id, deleted_chunks}` — efface **cette** organisation |
| `GET /admin/organizations` | — | `{organizations: [...], total, chargees}` |
| `DELETE /admin/organizations/{org}` | — | `{ok, deleted_documents, deleted_chunks}` — **supprime la base, répertoire compris** |

À l'ingestion, le document est découpé en fragments de 512 caractères avec 80 de
recouvrement, enrichi de métadonnées, encodé et indexé dans la base de cette
organisation — et d'aucune autre.

#### L'identifiant du document

`document_id` est **fourni par l'appelant** et conservé tel quel. Réutiliser le
même identifiant **remplace intégralement** le document : les fragments existants
sont supprimés, puis les nouveaux insérés. `replaced` dans la réponse dit si un
document a effectivement été remplacé.

Ce remplacement ne peut pas reposer sur la seule idempotence de l'indexation :
l'identifiant d'un fragment dérive de son texte, donc un contenu modifié produit
de nouveaux fragments qui s'ajouteraient **à côté** des anciens. La suppression
préalable est ce qui évite deux versions contradictoires de la même démarche
dans l'index.

L'opération n'est pas atomique : il existe une fenêtre de quelques secondes
pendant laquelle le document est absent de l'index.

Sans `document_id`, un identifiant est dérivé du titre et du début du texte —
repli destiné aux appels manuels, à ne pas utiliser pour un document qu'on
republiera.

#### Ingestion par pointeur (MinIO)

```bash
curl -X POST "http://localhost:8008/admin/organizations/$ORG/documents/from-storage" \
  -H "Content-Type: application/json" \
  -d '{"documentId":"d4c1f0a2-8e35-4b77-9a10-5f3c2e8b1d69",
       "bucket":"tontuma-documents",
       "objectKey":"'"$ORG"'/dossier-medical-v3.pdf",
       "title":"Dossier médical","category":"procedure"}'
```

Le service lit l'objet en **lecture seule** et ne lui écrit jamais. Nécessite
`S3_ENDPOINT`, `S3_ACCESS_KEY`, `S3_SECRET_KEY` et `S3_PATH_STYLE_ACCESS=true`
(obligatoire pour MinIO). Non configuré → `500` ; configuré mais injoignable →
`502`.

#### Ingestion directe

```bash
curl -X POST "http://localhost:8008/admin/organizations/$ORG/documents" \
  -F "file=@procedures.pdf" -F "title=Procédures hospitalières" \
  -F "document_id=d4c1f0a2-8e35-4b77-9a10-5f3c2e8b1d69" -F "category=procedure"
```

#### Isolation

Une organisation est créée à la première ingestion ; **les lectures n'en créent
aucune**. Sa base vit dans `data/chroma/{organization_id}/`, avec un `meta.json`
qui la décrit (modèle d'embedding, dates, comptages). `DELETE
/admin/organizations/{org}` efface ce répertoire : il ne reste rien sur le
disque.

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
  body: JSON.stringify({ question, organization_id: organizationId,
                         tts: true, session_id: sessionId }),
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
    .bodyValue(Map.of("question", question, "organization_id", organizationId,
                      "tts", true, "session_id", sessionId))
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
  -d '{"question":"Test","organization_id":"9b7d1e44-2c08-4a17-b3f5-6e2d9c447a10","tts":false}'
```

`-N` désactive la mise en tampon de curl, sans quoi le flux n'apparaît qu'à la fin.

---

## Codes d'erreur

| Code | Quand |
|---|---|
| `200` | y compris pour un flux qui se terminera par `event: error` |
| `400` | question vide, `organization_id` absent ou mal formé, format refusé, texte vide, PDF sans texte extractible, PDF illisible |
| `404` | document inconnu à la suppression, organisation inconnue à la suppression, objet introuvable sur le stockage |
| `409` | l'index a été construit avec un autre `EMBED_MODEL` — **aucune donnée n'est touchée**, une réindexation est requise |
| `500` | erreur interne d'ingestion, ou stockage objet non configuré |
| `502` | stockage objet configuré mais injoignable |

**Une erreur de pipeline n'est pas un code HTTP.** Le flux ayant déjà commencé
avec un `200`, l'échec arrive dans `event: error` puis `event: done`. Traitez
les deux.
