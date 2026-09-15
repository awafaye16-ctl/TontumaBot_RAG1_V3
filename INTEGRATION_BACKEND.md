# TontumaBot — Intégration Backend ↔ IA

> Contrat d'intégration entre le **backend Spring Boot** (+ MinIO) et le
> **service IA** (FastAPI, RAG multilingue Wolof/Français).
>
> **Version du contrat** : 3.1 · **Service IA** : `3.1.0` — côté IA, implémenté
> **Public** : équipe backend. Pour le détail interne du pipeline IA, voir `README.md` ;
> pour le contrat brut de l'API, voir `API.md`.

---

## Table des matières

1. [Statut : ce qui existe, ce qui arrive](#1-statut--ce-qui-existe-ce-qui-arrive)
2. [Architecture et responsabilités](#2-architecture-et-responsabilités)
3. [Isolation multi-tenant — le principe central](#3-isolation-multi-tenant--le-principe-central)
4. [Sécurité et réseau — à lire avant de déployer](#4-sécurité-et-réseau--à-lire-avant-de-déployer)
5. [Référence des endpoints](#5-référence-des-endpoints)
6. [Consommer le flux SSE depuis Spring Boot](#6-consommer-le-flux-sse-depuis-spring-boot)
7. [Ingestion documentaire via MinIO](#7-ingestion-documentaire-via-minio)
8. [Traçabilité : le champ `sources`](#8-traçabilité--le-champ-sources)
9. [Erreurs, délais et reprise](#9-erreurs-délais-et-reprise)
10. [Configuration](#10-configuration)
11. [Checklist d'intégration](#11-checklist-dintégration)
12. [Limites connues et évolutions](#12-limites-connues-et-évolutions)

---

## 1. Statut : ce qui existe, ce qui arrive

**Lisez ce tableau avant d'écrire une ligne de code.** Tout ce qui suit est
**implémenté et testé côté IA en 3.1.0**. Vous pouvez coder contre l'intégralité
de ce document.

| Sujet | Statut | Détail |
|---|---|---|
| `POST /ask` en SSE | ✅ disponible | inchangé |
| `POST /ask/audio` en SSE | ✅ disponible | inchangé |
| Mémoire conversationnelle (`session_id`) | ✅ disponible | RAM du process — voir §12 |
| `GET /health` | ✅ disponible | enrichi en 3.1 |
| Ingestion directe (`multipart`) | ✅ disponible | conservée |
| Traduction seule (`GET /translate`) | ✅ disponible | outil de test |
| **Isolation par organisation** | ✅ 3.1 | une base vectorielle par structure |
| **`organization_id` sur `/ask` et `/ask/audio`** | ✅ 3.1 | obligatoire |
| **Endpoints admin scopés par organisation** | ✅ 3.1 | `organization_id` dans le chemin |
| **`document_id` renvoyé à l'ingestion** | ✅ 3.1 | + id fourni par l'appelant |
| **Champ `sources` dans `result`** | ✅ 3.1 | traçabilité RAG |
| **Ingestion par pointeur MinIO** | ✅ 3.1 | `bucket` + `objectKey` |
| Authentification | ❌ absente | **voir §4 — c'est structurant** |

> **Ce qui reste à faire, et de quel côté.** Côté IA, rien : les six points sont
> livrés et couverts par des tests d'isolation (une question de l'organisation A
> ne peut pas citer un document de B ; `clear` sur A laisse B intacte ; une
> organisation vide ne se rabat sur rien). Côté déploiement, deux choses
> conditionnent la mise en production et ne dépendent pas de nous : l'isolation
> réseau du service IA (§4) et le compte MinIO en lecture seule (§7).

---

## 2. Architecture et responsabilités

```
   Usager (navigateur, borne tactile)
                │
                ▼
   ┌────────────────────────────┐
   │   Backend Spring Boot      │  ← authentification, organisations,
   │   :8080  (exposé)          │    sessions, droits, journalisation métier
   └───────┬─────────────┬──────┘
           │             │
   dépose  │             │  HTTP privé (jamais exposé publiquement)
   le      │             ▼
  fichier  │   ┌────────────────────────────┐
           │   │   Service IA FastAPI       │  ← STT, traduction, RAG,
           │   │   :8008  (réseau privé)    │    LLM, TTS
           │   └───────┬────────────────┬───┘
           │           │                │
           ▼           │ lit le fichier │ une base vectorielle
   ┌──────────────┐    │ à l'indexation │ par organisation
   │    MinIO     │◄───┘                ▼
   │  (S3-compat) │          data/chroma/{organization_id}/
   └──────────────┘
```

### Qui fait quoi

| Responsabilité | Backend | IA |
|---|---|---|
| Authentifier l'usager, résoudre son organisation | ✅ | ❌ |
| Gérer le cycle de vie des organisations | ✅ | ❌ |
| Stocker le fichier source (PDF, TXT, MD) | ✅ MinIO | ❌ |
| Générer et conserver le `documentId` | ✅ | ❌ |
| Extraire, découper, vectoriser, indexer | ❌ | ✅ |
| Isoler les bases entre organisations | ❌ | ✅ |
| Générer et suivre le `session_id` | ✅ | ❌ |
| Comprendre la question, produire la réponse | ❌ | ✅ |
| Relayer l'audio vers le navigateur | ✅ (proxy) | ✅ (produit) |
| Exposer quoi que ce soit à Internet | ✅ | ❌ **jamais** |

**Le principe** : l'IA est un service de calcul sans notion d'identité ni de
droits. Elle fait confiance à l'`organization_id` qu'on lui transmet. C'est le
backend qui garantit que cet identifiant est légitime.

---

## 3. Isolation multi-tenant — le principe central

TontumaBot est un SaaS : plusieurs structures (mairies, hôpitaux, préfectures)
partagent la plateforme, chacune avec sa base de connaissances **strictement
isolée**.

### Le choix d'implémentation retenu

**Une base vectorielle physiquement séparée par organisation**, sur le disque du
service IA :

```
data/chroma/
  ├── 3f2a8c10-7b41-4e9d-9c22-1a5e6d0b8f31/     ← mairie de Thiès
  │   ├── chroma.sqlite3
  │   └── meta.json        {organization_id, nom, embed_model, créé_le, n_chunks}
  └── 9b7d1e44-2c08-4a17-b3f5-6e2d9c447a10/     ← hôpital Fann
      ├── chroma.sqlite3
      └── meta.json
```

**Pourquoi ce choix plutôt qu'un filtre de métadonnées** : le moteur de recherche
hybride de l'IA charge son corpus en mémoire et construit un index lexical BM25
dessus — un filtre applicatif devrait être réappliqué correctement sur chaque
chemin de recherche, et **un seul oubli produirait une fuite silencieuse** entre
structures. Avec des bases séparées, l'isolation est structurelle : une requête
sur la base A ne *peut pas* retourner un document de B. Bénéfices annexes pour
vous : suppression d'une structure = suppression d'un répertoire, sauvegarde et
restauration par structure, et migration possible vers un déploiement sur site.

### Ce que ça implique pour le backend

**1. `organization_id` est un UUID, et il est obligatoire.**
Le service IA le refuse en `400` s'il est absent ou mal formé. Aucune valeur par
défaut, aucun repli sur une base « globale ».

**2. Sur les endpoints d'administration, il est dans le chemin.**
Ce n'est pas cosmétique : un segment de chemin oublié donne un `404`, alors qu'un
champ de corps oublié donnerait une opération silencieusement globale. Sur une
API sans authentification, la différence entre un `404` et « effacer les
documents de tous les clients » vaut la verbosité de l'URL.

```
POST   /admin/organizations/{organization_id}/documents
GET    /admin/organizations/{organization_id}/documents
DELETE /admin/organizations/{organization_id}/documents/{document_id}
POST   /admin/organizations/{organization_id}/documents/clear
DELETE /admin/organizations/{organization_id}
```

**3. Une organisation inconnue est créée à la volée.**
Le premier document ingéré pour un `organization_id` crée sa base. Il n'y a pas
d'endpoint « créer une organisation » : c'est le backend qui détient le registre.
Corollaire : **une faute de frappe dans l'UUID crée une base orpheline** au lieu
de lever une erreur. Validez côté backend avant l'appel.

**4. Interroger une organisation sans documents ne renvoie pas d'erreur.**
Le pipeline répond qu'il ne dispose pas d'information, sans inventer et **sans
se rabattre sur les documents d'une autre structure ni sur un jeu de démonstration**.

---

## 4. Sécurité et réseau — à lire avant de déployer

**Le service IA n'a aucune authentification.** Tous les endpoints sont publics,
CORS est ouvert à `*`, et le fichier `api.key` présent dans le dépôt n'est lu par
aucune route. Ce n'est pas un oubli à corriger plus tard : c'est une contrainte
de déploiement à intégrer maintenant.

Conséquence directe : **l'`organization_id` est une clé de routage, pas une
frontière de sécurité.** Qui peut joindre le service IA peut envoyer n'importe
quel `organization_id` — donc lire et effacer les documents de n'importe quelle
structure.

### Les trois règles à respecter

**1. Le service IA n'est jamais joignable depuis Internet.**
En Docker Compose, ne publiez pas son port : laissez-le sur le réseau interne et
ne mettez `ports:` que sur le backend.

```yaml
services:
  backend:
    ports: ["8080:8080"]          # exposé
    networks: [tontuma]
  ia:
    expose: ["8008"]              # visible du backend seulement, PAS de `ports:`
    networks: [tontuma]
  minio:
    expose: ["9000"]
    networks: [tontuma]
networks:
  tontuma:
    driver: bridge
```

**2. Le backend est le seul client de l'IA.**
Aucun appel direct du navigateur vers `:8008`, y compris pour l'audio (voir
règle 3). Si une borne tactile doit parler à l'IA, elle passe par le backend.

**3. L'audio doit être relayé par le backend.**
Les réponses SSE portent des URL relatives (`/static/response_ab12cd34.wav`). Le
navigateur ne pouvant pas joindre l'IA, le backend doit exposer un proxy :

```
GET /api/v1/chat/audio/{fichier}  →  GET http://ia:8008/static/{fichier}
```

Validez le nom de fichier contre `^response_[0-9a-f]{8}(_\d{2})?\.wav$` avant de
relayer : sans cela, ce proxy sert de lecteur de fichiers arbitraires sur le
disque de l'IA.

> **Si vous décidez d'ajouter un secret partagé** (en-tête `X-Internal-Token`
> entre backend et IA), dites-le à l'équipe IA : c'est une dizaine de lignes de
> middleware FastAPI. Ça ne remplace pas l'isolation réseau, ça la double.

---

## 5. Référence des endpoints

**Base URL** : `http://ia:8008` (nom du service sur le réseau interne).
Toutes les durées sont en millisecondes.

---

### `GET /health`

Sonde de disponibilité, sans effet de bord. À utiliser pour le `healthcheck`
Docker et le circuit breaker Spring.

```json
{
  "status": "ok",
  "version": "3.1.0",
  "llm_provider": "groq",
  "llm_ready": true,
  "tts": "oolel-voices",
  "device": { "auto": "mps", "cuda": false, "mps": true },
  "organizations": { "total": 12, "chargees": 3 },
  "memory": { "enabled": true, "max_messages": 10, "sessions": 8 }
}
```

- `llm_ready: false` → aucune clé LLM configurée. **Le service répond quand
  même**, en renvoyant le passage documentaire le plus pertinent au lieu d'une
  réponse rédigée. Ne traitez pas ça comme une panne, mais remontez-le.
- `device.auto: "cpu"` alors que du GPU est attendu → les latences triplent.
  C'est la première chose à regarder si le service paraît lent.
- `organizations.chargees` = bases actuellement en mémoire (cache LRU).

> En 3.0.0, ce document expose `n_documents` et `n_chunks` globaux ; ils
> disparaissent en 3.1 au profit de `organizations`, les comptages devenant
> propres à chaque structure.

---

### `POST /ask` — question écrite

`Content-Type: application/json` → **réponse SSE** (`text/event-stream`).

| Champ | Type | Requis | Défaut | Description |
|---|---|---|---|---|
| `question` | string | **oui** | — | en wolof ou en français |
| `organization_id` | UUID | **oui** (3.1) | — | détermine la base de connaissances interrogée |
| `session_id` | string | non | `null` | votre identifiant de conversation |
| `lang` | string | non | `null` | `wo` \| `fr` — sinon détection automatique |
| `tts` | bool | non | `false` | produire aussi l'audio |
| `provider` | string | non | `.env` | `groq` \| `gemini` \| `local` |

Sur ce chemin, **omettre `lang` est le comportement normal** : la détection
travaille sur du texte, où elle est fiable. Ne le transmettez que si l'usager a
explicitement choisi une langue dans l'interface.

```json
{
  "question": "Comment obtenir une copie de mon dossier médical ?",
  "organization_id": "9b7d1e44-2c08-4a17-b3f5-6e2d9c447a10",
  "session_id": "b3f5a1c2-...",
  "tts": true
}
```

---

### `POST /ask/audio` — question vocale

`multipart/form-data` → **réponse SSE**.

| Champ | Type | Requis | Défaut | Description |
|---|---|---|---|---|
| `file` | fichier | **oui** | — | WAV, MP3, M4A, WebM |
| `organization_id` | UUID | **oui** (3.1) | — | base de connaissances interrogée |
| `lang` | string | non | `wo` | **choisit le moteur de transcription** |
| `session_id` | string | non | `null` | identifiant de conversation |
| `tts` | bool | non | `false` | produire aussi l'audio |
| `provider` | string | non | `.env` | `groq` \| `gemini` \| `local` |

> **`lang` n'est pas optionnel en pratique.** Il sélectionne le moteur de
> transcription — `wo` → Wolof-HuBERT-CTC, `fr` → Whisper — et ce choix se fait
> *avant* qu'il existe une transcription à analyser : la langue d'un fichier
> audio n'est pas détectable à ce stade. Le défaut `wo` signifie qu'une question
> posée en français **sans** ce champ sera transcrite par le modèle wolof, puis
> répondue en wolof. Faites déclarer la langue par l'usager et transmettez-la.

---

### Les événements du flux SSE

Format : `event: <nom>\ndata: <json>\n\n`. Un flux se termine **toujours** par
`done`, y compris après une erreur.

```
start → (stt) → detect → (alerte_langue) → (translate_in) → intent
      → retrieval → llm → (translate_out) → (alerte_nombres)
      → (tts) → (tts_chunk × N) → result → done
```

Les étapes entre parenthèses n'apparaissent que si elles ont lieu.
**Traitez toute étape inconnue comme ignorable** : de nouvelles peuvent apparaître
sans préavis, et votre client ne doit pas casser pour autant.

| Événement | `step` | Ce qu'il porte | À quoi ça vous sert |
|---|---|---|---|
| `status` | `start` | `question`, `provider`, `tts` | requête acceptée ; sur `/ask/audio`, `question` porte la transcription |
| `status` | `stt` | `lang` | transcription en cours |
| `status` | `detect` | `lang`, `source` | `source` = `indice` (déclarée) ou `détection` (déduite) |
| `status` | `intent` | `intent` | `procedure` ou `orientation` |
| `status` | `retrieval` | — | recherche documentaire |
| `status` | `llm` | `provider` | rédaction en cours |
| `status` | **`tts`** | `response`, `response_fr`, `response_wo`, `lang` | **le texte de la réponse, disponible avant l'audio** |
| `status` | `tts_chunk` | `index`, `total`, `audio_url`, `latency_ms` | un morceau audio lisible immédiatement |
| `status` | `alerte_langue` | `declaree`, `detectee`, `scores` | la transcription contredit la langue annoncée |
| `status` | `alerte_nombres` | `controles`, `suspects`, `coherent` | un montant s'est perdu à la traduction |
| `result` | — | la réponse complète (ci-dessous) | |
| `error` | — | `message` | suivi de `done` |
| `done` | — | `{}` | fin du flux |

#### Deux événements à ne pas ignorer

**`status: tts` porte le texte.** Le `result` n'arrive qu'une fois l'audio
produit — jusqu'à deux minutes plus tard sur une réponse longue. Relayez le texte
dès cet événement, sinon votre interface parlera avant d'afficher ce qu'elle dit.

**`status: alerte_langue`** signale qu'un usager s'est trompé de bouton. **Rien
n'est corrigé** : le mal est fait à la transcription et la langue déclarée
continue de gouverner tout le pipeline. L'événement existe pour que vous puissiez
l'expliquer à l'usager, dont la réponse va sembler absurde. Ne l'avalez pas
silencieusement.

#### L'événement `result`

```json
{
  "response":    "Ngir mu mën a dem lopitaal bi...",
  "response_fr": "Pour se rendre à l'hôpital...",
  "response_wo": "Ngir mu mën a dem...",
  "qr_code":     "iVBORw0KGgo...",
  "audio_url":   "/static/response_ab12cd34.wav",
  "lang":        "wo",
  "sources":     [ ... ],
  "memory":      { "count": 4, "max": 10, "reset": false },
  "trace":       { ... }
}
```

| Champ | Toujours présent | Remarque |
|---|---|---|
| `response` | ✅ | dans la langue de l'usager |
| `response_fr` | ✅ | pivot du pipeline |
| `response_wo` | ❌ | absent si la question était en français |
| `qr_code` | ❌ | PNG en base64, sur les procédures seulement |
| `audio_url` | ❌ | `null` si `tts: false` — **à relayer via votre proxy** |
| `lang` | ✅ | langue retenue |
| `sources` | ✅ | traçabilité — voir §8 |
| `memory` | ❌ | `null` si aucun `session_id` |
| `trace` | ✅ | observabilité — volumineux, voir §9 |
| `confidence` | ❌ **jamais** | le pipeline n'en produit pas ; traitez-le comme absent |

**`memory.reset: true` signifie que l'historique vient d'être vidé** : la question
suivante ne bénéficiera plus du contexte. Prévenez l'usager, sinon ses questions
de suite (« et combien ça coûte ? ») cesseront d'être comprises sans explication.

---

### Mémoire conversationnelle

Transmettez le même `session_id` d'une question à l'autre. L'IA retient
l'historique **en français** (langue pivot) et s'en sert pour comprendre les
questions de suite. L'historique n'est **jamais** une source de faits : seule la
base documentaire fonde une réponse.

Au-delà de `MEMORY_MAX_MESSAGES` (10 par défaut, soit 5 échanges), la mémoire est
**vidée** — la session continue, aucune erreur n'est levée. C'est `memory.reset`
dans `result` qui vous en informe.

| Endpoint | Effet |
|---|---|
| `GET /session/{id}` | `{session_id, messages, count, max, resets, total}` — **jamais 404**, une session inconnue renvoie un état vide |
| `POST /session/{id}/reset` | vide l'historique, garde la session → `{ok, existed}` |
| `DELETE /session/{id}` | oublie la session → `{ok, existed}` |

> ⚠️ **Ces endpoints ne sont pas scopés par organisation et ne le seront pas.**
> Qui connaît un `session_id` lit l'historique complet de la conversation. Deux
> conséquences : générez des UUID v4 (jamais d'identifiants séquentiels ou
> devinables), et n'exposez jamais ces routes au travers de votre proxy public.

---

### Administration documentaire

Tous ces endpoints sont scopés par organisation. Aucune authentification — voir §4.

#### `POST /admin/organizations/{organization_id}/documents/from-storage`

**Voie recommandée.** Le backend dépose le fichier sur MinIO puis notifie l'IA
avec un pointeur ; l'IA lit l'objet et l'indexe. Détail du flux en §7.

`Content-Type: application/json`

| Champ | Type | Requis | Description |
|---|---|---|---|
| `documentId` | string | **oui** | **votre** identifiant — réutilisez-le pour remplacer |
| `bucket` | string | **oui** | bucket MinIO |
| `objectKey` | string | **oui** | clé de l'objet |
| `title` | string | non | titre affiché ; défaut = nom de l'objet |
| `category` | string | non | métadonnée libre, conservée et renvoyée |

```json
{
  "documentId": "d4c1f0a2-8e35-4b77-9a10-5f3c2e8b1d69",
  "bucket": "tontuma-documents",
  "objectKey": "9b7d1e44-2c08-4a17-b3f5-6e2d9c447a10/dossier-medical-v3.pdf",
  "title": "Demande de copie de dossier médical",
  "category": "procedure"
}
```

Réponse `200` :

```json
{
  "ok": true,
  "document_id": "d4c1f0a2-8e35-4b77-9a10-5f3c2e8b1d69",
  "organization_id": "9b7d1e44-2c08-4a17-b3f5-6e2d9c447a10",
  "title": "Demande de copie de dossier médical",
  "chunks": 32,
  "replaced": true,
  "added": "2026-09-14T11:42:07Z"
}
```

`replaced: true` indique qu'un document portant ce `documentId` existait et a été
**entièrement remplacé**. Voir §7 pour ce que ça garantit exactement.

#### `POST /admin/organizations/{organization_id}/documents`

Ingestion directe, conservée pour les tests et les petits contenus.
`multipart/form-data` : `file` (`.txt`, `.md`, `.pdf`) **ou** `text`, plus
`title` et `document_id` optionnels. Même forme de réponse.

```bash
curl -X POST "http://ia:8008/admin/organizations/$ORG/documents" \
  -F "file=@procedures.pdf" -F "title=Procédures hospitalières"
```

#### `GET /admin/organizations/{organization_id}/documents`

```json
{
  "organization_id": "9b7d1e44-...",
  "documents": [
    { "id": "d4c1f0a2-...", "title": "Demande de copie...",
      "source": "dossier-medical-v3.pdf", "category": "procedure",
      "chunks": 32, "added": "2026-09-14T11:42:07Z" }
  ],
  "total_documents": 1,
  "total_chunks": 32
}
```

Une organisation inconnue renvoie une liste vide, **pas un 404** : elle n'existe
simplement pas encore.

#### `DELETE /admin/organizations/{organization_id}/documents/{document_id}`

`{"ok": true, "deleted_chunks": 32}` · `404` si le document est inconnu **dans
cette organisation**.

#### `POST /admin/organizations/{organization_id}/documents/clear`

Efface tous les documents de **cette** organisation. `{"ok": true, "deleted_chunks": 32}`

#### `DELETE /admin/organizations/{organization_id}`

Supprime la base vectorielle de la structure — répertoire compris. Irréversible.
À appeler à la résiliation d'un client. `{"ok": true, "deleted_chunks": 32, "deleted_documents": 1}`
`404` si l'organisation n'a jamais eu de base.

#### `GET /admin/organizations`

Supervision : les organisations présentes sur le disque de l'IA, avec leur date
de création, leur modèle d'embedding et leurs comptages.

```json
{"organizations": [
   {"organization_id": "9b7d1e44-...", "cree_le": "2026-09-14T11:42:07Z",
    "maj_le": "2026-09-15T08:10:22Z", "n_documents": 12, "n_chunks": 341,
    "embed_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"}],
 "total": 1, "chargees": 1}
```

Utile pour rapprocher votre registre d'organisations de ce que l'IA détient
réellement. `chargees` est le nombre de bases actuellement en mémoire (cache LRU).

> **Les lectures ne créent rien.** Interroger `GET …/documents` sur une
> organisation inconnue renvoie une liste vide **sans créer sa base** : une
> sonde, un balayage ou une faute de frappe ne laissent aucune trace sur le
> disque de l'IA. Seule une ingestion crée une organisation.

---

## 6. Consommer le flux SSE depuis Spring Boot

Le point d'attention n'est pas le protocole — c'est que **la chaîne complète dure
de 6 à 120 secondes**. Les valeurs par défaut de WebClient ne sont pas
compatibles avec ça, et vous obtiendrez des `ReadTimeoutException` en production
sur les réponses longues alors que tout fonctionnait en développement.

### Configuration du client — les quatre réglages obligatoires

```java
@Configuration
public class IaClientConfig {

    @Bean
    public WebClient iaWebClient(@Value("${tontuma.ia.base-url}") String baseUrl) {

        HttpClient httpClient = HttpClient.create()
            .option(ChannelOption.CONNECT_TIMEOUT_MILLIS, 5_000)
            // 1. PAS de responseTimeout global : le flux dure jusqu'à 2 min.
            //    On borne l'inactivité ENTRE deux événements, pas la durée totale.
            .doOnConnected(conn -> conn.addHandlerLast(
                new ReadTimeoutHandler(180, TimeUnit.SECONDS)));

        return WebClient.builder()
            .baseUrl(baseUrl)
            .clientConnector(new ReactorClientHttpConnector(httpClient))
            // 2. Le `result` peut dépasser le plafond par défaut de 256 Ko :
            //    un QR code en base64 + la trace complète pèsent lourd.
            .codecs(c -> c.defaultCodecs().maxInMemorySize(8 * 1024 * 1024))
            .build();
    }
}
```

Les deux autres réglages :

**3. Ne mettez pas ce client derrière un timeout de circuit breaker court.**
Resilience4j avec un `timeLimiter` à 10 s coupera systématiquement les réponses
vocales. Si vous en voulez un, réglez-le à 180 s ou excluez ce chemin.

**4. Si un reverse proxy se trouve entre backend et IA**, désactivez la mise en
tampon des réponses, sinon le flux n'arrivera qu'à la fin et vous perdez tout
l'intérêt du streaming. L'IA émet déjà `X-Accel-Buffering: no` pour nginx.

### Consommation du flux

```java
public Flux<ChatEvent> ask(AskRequest req) {
    return iaWebClient.post()
        .uri("/ask")
        .contentType(MediaType.APPLICATION_JSON)
        .bodyValue(req)
        .accept(MediaType.TEXT_EVENT_STREAM)
        .retrieve()
        .bodyToFlux(new ParameterizedTypeReference<ServerSentEvent<String>>() {})
        .mapNotNull(sse -> {
            String type = sse.event();      // "status" | "result" | "error" | "done"
            String data = sse.data();
            if (type == null || data == null) return null;

            return switch (type) {
                case "status" -> {
                    JsonNode n = mapper.readTree(data);
                    String step = n.path("step").asText();
                    // Le texte arrive ICI, bien avant `result`.
                    if ("tts".equals(step))       yield ChatEvent.texte(n);
                    if ("tts_chunk".equals(step)) yield ChatEvent.morceauAudio(n);
                    if ("alerte_langue".equals(step)) yield ChatEvent.alerte(n);
                    yield ChatEvent.progression(step);   // étape inconnue incluse
                }
                case "result" -> ChatEvent.resultat(mapper.readValue(data, ResultatIa.class));
                case "error"  -> ChatEvent.erreur(mapper.readTree(data).path("message").asText());
                case "done"   -> null;   // fin de flux, rien à relayer
                default       -> null;   // événement futur : on ignore, on ne casse pas
            };
        })
        .timeout(Duration.ofMinutes(4))   // garde-fou global
        .onErrorResume(e -> Flux.just(ChatEvent.erreur("Service IA indisponible")));
    }
```

### Les DTO

Déclarez `@JsonIgnoreProperties(ignoreUnknown = true)` sur **tous** les DTO : le
service IA ajoute des champs sans changer de version majeure.

```java
@JsonIgnoreProperties(ignoreUnknown = true)
public record ResultatIa(
    String response,
    @JsonProperty("response_fr") String responseFr,
    @JsonProperty("response_wo") String responseWo,   // null si question en français
    @JsonProperty("qr_code")     String qrCodeBase64, // null hors procédures
    @JsonProperty("audio_url")   String audioUrl,     // null si tts=false
    String lang,
    List<Source> sources,
    Memoire memory,
    JsonNode trace                                    // ne le modélisez pas, voir §9
) {}

@JsonIgnoreProperties(ignoreUnknown = true)
public record Source(
    @JsonProperty("document_id") String documentId,
    String title,
    String category,
    @JsonProperty("chunk_id") String chunkId,
    int rank,
    double score
) {}
```

### Trois pièges vérifiés en pratique

**Ne découpez jamais le flux vous-même.** Si vous consommez en `bodyToFlux(String.class)`
plutôt qu'en `ServerSentEvent`, un événement peut arriver coupé en deux lectures
réseau. Le découpage doit se faire sur `\n\n` **en conservant le fragment
incomplet**. Utilisez `ServerSentEvent` et laissez Spring s'en charger.

**Un `200` ne signifie pas que ça a marché.** Le flux commence par un `200` avant
que le pipeline ait travaillé : un échec arrive dans `event: error` puis
`event: done`, pas dans le code HTTP. Traitez les deux.

**La synthèse vocale produit environ deux fois moins vite qu'on n'écoute.** Votre
file de lecture se videra en cours de route. Le client (navigateur, borne) doit
rester en attente du morceau suivant plutôt que de rendre la main. Sur une
réponse en 5 morceaux, le premier son est disponible à 37 s contre 153 s pour
l'énoncé complet — c'est tout l'intérêt de relayer les `tts_chunk`.

---

## 7. Ingestion documentaire via MinIO

### Le flux

```
1. Un agent dépose un PDF dans l'interface d'administration du backend
2. Backend  → génère un documentId (UUID v4), le persiste en base
3. Backend  → PUT sur MinIO : {bucket}/{organizationId}/{nom-fichier}
4. Backend  → POST /admin/organizations/{orgId}/documents/from-storage
                  { documentId, bucket, objectKey, title, category }
5. IA       → GET sur MinIO, extrait le texte, découpe, vectorise, indexe
6. IA       → 200 { document_id, chunks, replaced, added }
7. Backend  → marque le document INDEXÉ en base, avec le nombre de chunks
```

### Convention de nommage des objets

**Préfixez toujours par l'`organizationId`** :

```
tontuma-documents/{organizationId}/{documentId}-{nom-original}.pdf
```

Ça vous donne trois choses gratuitement : une politique IAM MinIO par préfixe si
vous en ajoutez une, un `mc ls` lisible pour le support, et une suppression de
structure en une commande.

### Remplacement d'un document — ce qui est garanti

Réutiliser le même `documentId` **remplace intégralement** le document : l'IA
supprime tous les fragments existants portant cet identifiant, puis réindexe.
C'est une suppression suivie d'une insertion, pas une fusion.

**Deux conséquences à connaître :**

1. **Il existe une fenêtre de quelques secondes pendant laquelle le document est
   absent de l'index.** Une question posée à cet instant précis ne le trouvera
   pas. C'est acceptable pour une republication de procédure administrative ;
   ne construisez pas de logique qui suppose une atomicité.

2. **Ne réutilisez jamais un `documentId` pour un document différent.** L'ancien
   contenu disparaîtra sans avertissement. Un `documentId` désigne *une*
   démarche administrative pour toute sa durée de vie ; ses versions successives
   partagent l'identifiant, deux démarches distinctes n'en partagent jamais.

### Formats et limites

| | |
|---|---|
| Formats acceptés | `.txt`, `.md`, `.pdf` |
| PDF | texte extractible uniquement — **un PDF scanné sans OCR produira 0 chunk** |
| Découpage | fragments de 512 caractères, 80 de recouvrement, sur frontières de phrases |
| Fragment minimal | 40 caractères — en dessous, il est écarté |

**Le cas du PDF scanné mérite un traitement explicite côté backend.** L'IA
renverra `400` avec `"Aucun texte exploitable dans ce fichier"` — c'est le cas le
plus fréquent en production, et l'agent qui a déposé le fichier doit comprendre
que son document est une image, pas que « ça a planté ».

### Durée et mode d'appel

L'indexation est **synchrone** : l'IA ne répond qu'une fois le document vectorisé.
Comptez de 2 à 30 secondes selon la taille (l'encodage des fragments domine).
Réglez le timeout de ce client à **120 s**, distinct de celui du chat.

```java
public IngestionResponse indexer(UUID orgId, IngestionRequest req) {
    return iaWebClient.post()
        .uri("/admin/organizations/{orgId}/documents/from-storage", orgId)
        .bodyValue(req)
        .retrieve()
        .onStatus(HttpStatusCode::is4xxClientError, r ->
            r.bodyToMono(String.class).map(IngestionRefusee::new))
        .bodyToMono(IngestionResponse.class)
        .timeout(Duration.ofSeconds(120))
        .block();
}
```

> Sur des documents de plusieurs centaines de pages, ce modèle synchrone
> atteindra ses limites. Un mode asynchrone (`202 Accepted` + endpoint de suivi)
> est prévu — voir §12. Dites-nous si vous rencontrez le cas avant.

### Reprise sur erreur

**Rejouer un appel d'ingestion est sans danger** : le remplacement par
`documentId` rend l'opération idempotente. Un appel répété produit le même état
final, pas des doublons.

| Situation | Ce que fait l'IA | Ce que doit faire le backend |
|---|---|---|
| Objet MinIO introuvable | `404` + message | Vérifier que le PUT a bien eu lieu **avant** le POST |
| MinIO injoignable | `502` | Rejouer avec backoff exponentiel |
| PDF sans texte extractible | `400` | Marquer `ÉCHEC_EXTRACTION`, alerter l'agent |
| Format non supporté | `400` | Refuser plus tôt, côté interface |
| Erreur interne d'indexation | `500` | Rejouer une fois, puis alerter |
| Timeout côté backend | *(l'indexation continue)* | **Ne pas rejouer immédiatement** : interroger `GET .../documents` pour savoir si elle a abouti |

### Configuration MinIO à transmettre à l'équipe IA

L'IA a besoin d'un accès **lecture seule** au bucket. Fournissez :

```properties
S3_ENDPOINT=http://minio:9000
S3_ACCESS_KEY=...              # compte dédié à l'IA, lecture seule
S3_SECRET_KEY=...
S3_REGION=us-east-1            # valeur arbitraire, MinIO l'ignore
S3_PATH_STYLE_ACCESS=true      # OBLIGATOIRE pour MinIO
```

> `S3_PATH_STYLE_ACCESS=true` n'est pas optionnel. Les clients S3 utilisent par
> défaut l'adressage par sous-domaine (`bucket.host/objet`), que MinIO ne sert
> pas. Sans ce réglage, tous les téléchargements échouent en résolution DNS —
> avec un message qui ne pointe pas du tout vers la vraie cause.

**Créez un compte de service dédié à l'IA**, avec une politique restreinte au
bucket documentaire et en lecture seule. L'IA n'écrit jamais sur MinIO.

---

## 8. Traçabilité : le champ `sources`

`result.sources` liste les documents qui ont effectivement nourri la réponse,
classés par pertinence décroissante.

```json
"sources": [
  { "document_id": "d4c1f0a2-8e35-4b77-9a10-5f3c2e8b1d69",
    "title": "Demande de copie de dossier médical",
    "category": "procedure",
    "chunk_id": "doc-3b5f134a2c8e9d01",
    "rank": 1,
    "score": 7.42 },
  { "document_id": "a91b7c30-...", "title": "Horaires des services",
    "category": null, "chunk_id": "doc-8e21f0c4b7a3d519",
    "rank": 2, "score": 2.18 }
]
```

| Champ | Ce que c'est |
|---|---|
| `document_id` | **l'identifiant que vous avez fourni à l'ingestion** — votre clé de jointure |
| `title` | titre du document |
| `category` | la catégorie que vous aviez transmise, ou `null` |
| `chunk_id` | identifiant du *fragment* — pour le débogage, pas pour la jointure |
| `rank` | 1 = le plus pertinent |
| `score` | score du reranker — **lisez l'avertissement ci-dessous** |

### Trois choses à comprendre avant d'afficher ça à un usager

**1. `score` n'est pas un pourcentage de confiance.** C'est la sortie brute d'un
cross-encoder : non bornée, souvent négative, non comparable d'une question à
l'autre. Elle donne un **ordre**, pas une note. Si vous voulez afficher une
pertinence, utilisez `rank`, ou faites un seuil que vous calibrez sur vos propres
données — mais ne présentez jamais ce nombre tel quel à un usager.

**2. Ne confondez pas `chunk_id` et `document_id`.** Le premier identifie un
fragment de 512 caractères et change à chaque réindexation ; le second est le
vôtre et reste stable. La jointure avec votre base se fait sur `document_id`.

**3. `sources` peut être vide.** Quand l'organisation n'a aucun document, ou
qu'aucun passage n'est pertinent, la liste est vide et la réponse indique que
l'information n'est pas disponible. **C'est un fonctionnement normal, pas une
erreur** — mais c'est un cas que votre interface doit savoir afficher, parce
qu'il sera fréquent sur une structure qui vient d'être créée.

Un même `document_id` peut apparaître plusieurs fois si deux de ses fragments ont
été retenus. Dédoublonnez à l'affichage si vous montrez des documents.

---

## 9. Erreurs, délais et reprise

### Codes HTTP

| Code | Quand | Réaction attendue |
|---|---|---|
| `200` | succès — **y compris pour un flux SSE qui se terminera par `event: error`** | — |
| `400` | question vide, `organization_id` absent ou mal formé, format refusé, PDF sans texte, PDF illisible | corriger l'appel ou prévenir l'agent |
| `404` | document inconnu dans cette organisation, objet MinIO introuvable | vérifier que le dépôt a précédé la notification |
| `409` | **l'index a été construit avec un autre modèle d'embedding** | alerter l'exploitation — réindexation requise |
| `500` | erreur interne d'ingestion, **ou stockage objet non configuré côté IA** | alerter, **ne pas rejouer** |
| `502` | MinIO injoignable depuis l'IA | rejouer avec backoff exponentiel |

### `409` — index incompatible

Une base construite avec un modèle d'embedding donné ne peut pas être lue avec
un autre : les vecteurs stockés ne sont plus comparables aux nouveaux. Quand
`EMBED_MODEL` change côté IA, le service **refuse d'ouvrir** les bases
concernées et répond `409`, plutôt que de les purger silencieusement.

Vous n'avez rien à coder pour ce cas au-delà d'une alerte : il ne peut survenir
qu'après une modification de configuration côté IA, et il se résout par une
réindexation, pas par un rejeu.

### Deux `400` à distinguer pour l'agent

Le message diffère parce que l'action de l'agent diffère :

| Message | Ce que ça veut dire | Ce que l'agent doit faire |
|---|---|---|
| « Aucun texte exploitable dans ce fichier… » | le PDF est valide mais ne contient qu'une image | faire passer le document par une reconnaissance de caractères |
| « PDF illisible (…). Le fichier est peut-être corrompu… » | le fichier est abîmé, tronqué ou protégé | redéposer le document |

Relayez le message tel quel plutôt qu'un « échec de l'indexation » générique :
c'est la différence entre un agent qui corrige son dépôt et un agent qui ouvre
un ticket.

**Une erreur de pipeline n'est pas un code HTTP.** Le flux ayant déjà commencé
avec un `200`, l'échec arrive dans `event: error` puis `event: done`. C'est la
source d'erreur d'intégration la plus fréquente : un client qui ne regarde que le
statut HTTP considère toutes les requêtes comme réussies.

### Latences observées (Apple M1, indicatif)

| Étape | Ordre de grandeur |
|---|---|
| Recherche documentaire | ~90 ms |
| Détection d'intention | ~50 ms |
| LLM | ~1 200 ms |
| Traduction FR→WO | ~3 300 ms |
| Traduction WO→FR | ~1 000 ms |
| **Synthèse vocale** | **~2 s par seconde d'audio produit** |

Une question en français sans TTS répond en 2 à 6 s. Une question en wolof avec
TTS peut atteindre 120 s. **Dimensionnez vos timeouts sur le pire cas, pas sur ce
que vous observez en développement en français.**

### Que faire de `trace`

`result.trace` porte le détail d'observabilité : latences par étape, candidats du
retrieval, scores du reranker, alertes. Elle est **volumineuse** (plusieurs Ko) et
sa structure **évoluera sans préavis**.

- **Ne la modélisez pas** en DTO typé — gardez un `JsonNode` ou une `Map`.
- **Ne la relayez jamais au navigateur** : elle contient des extraits de documents
  et l'intégralité du raisonnement de recherche.
- **Journalisez-la** côté backend, en `DEBUG`, avec le `session_id`. Le jour où un
  agent signale une réponse aberrante, c'est la seule chose qui permettra de dire
  si le problème vient du retrieval, de la traduction ou du LLM.

### Dégradations à anticiper

| Symptôme | Cause probable | Réaction |
|---|---|---|
| `llm_ready: false` sur `/health` | aucune clé LLM configurée | le service répond quand même, en mode extractif — alerter l'exploitation |
| Latences ×3 | `device.auto: "cpu"` | repli silencieux sur CPU, vérifier le matériel |
| `429` remonté dans `event: error` | quota du fournisseur LLM atteint | basculer `provider`, ou file d'attente |
| `alerte_nombres` fréquent | la traduction déforme les montants | à remonter à l'équipe IA avec le `session_id` |
| Réponses sans rapport | mauvais `organization_id` | vérifier la résolution d'organisation côté backend |

---

## 10. Configuration

### Côté backend (`application.yml`)

```yaml
tontuma:
  ia:
    base-url: http://ia:8008
    chat-timeout: 240s          # flux SSE : jusqu'à 2 min + marge
    ingestion-timeout: 120s     # indexation synchrone
    connect-timeout: 5s
  minio:
    endpoint: http://minio:9000
    bucket: tontuma-documents
```

### Côté IA (`.env`) — à fournir par l'exploitation

```properties
HOST=0.0.0.0
PORT=8008
LLM_PROVIDER=groq
GROQ_API_KEY=...

MEMORY_ENABLED=true
MEMORY_MAX_MESSAGES=10          # 10 entrées = 5 échanges
MEMORY_TTL_MINUTES=120

EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# Accès MinIO en lecture seule (3.1)
S3_ENDPOINT=http://minio:9000
S3_ACCESS_KEY=...
S3_SECRET_KEY=...
S3_PATH_STYLE_ACCESS=true

# Multi-tenant (3.1)
CHROMA_ROOT=./data/chroma
MAX_ORGANISATIONS_EN_CACHE=20   # éviction LRU des bases chargées
```

> ⚠️ **`EMBED_MODEL` ne se change pas à chaud.** Modifier cette valeur invalide
> **tous** les index de **toutes** les structures : les vecteurs déjà stockés
> deviennent incomparables aux nouveaux. Un changement impose une réindexation
> complète de chaque organisation, à planifier comme une opération de
> maintenance. Ne le touchez pas sans coordination avec l'équipe IA.

---

## 11. Checklist d'intégration

### Avant le premier appel

- [ ] Le service IA est sur le réseau interne, **sans `ports:` publié**
- [ ] `GET /health` répond depuis le conteneur backend
- [ ] `llm_ready: true` et `device.auto` conforme au matériel attendu
- [ ] Compte MinIO en lecture seule créé, credentials transmis à l'équipe IA
- [ ] `S3_PATH_STYLE_ACCESS=true` confirmé côté IA

### Chat

- [ ] `organization_id` (UUID v4) transmis sur **tous** les appels
- [ ] `session_id` généré en UUID v4, jamais séquentiel
- [ ] `maxInMemorySize` porté à 8 Mo sur le WebClient
- [ ] Timeouts relevés : connexion 5 s, lecture 180 s, global 240 s
- [ ] Aucun circuit breaker à timeout court sur ce chemin
- [ ] `event: error` traité — un `200` ne suffit pas
- [ ] Événement `status: tts` relayé pour afficher le texte **avant** l'audio
- [ ] Événements `status` inconnus ignorés sans lever d'exception
- [ ] `alerte_langue` remonté à l'interface
- [ ] `memory.reset: true` signalé à l'usager
- [ ] `trace` journalisée en DEBUG, **jamais** renvoyée au navigateur
- [ ] Proxy audio en place, nom de fichier validé par expression régulière

### Ingestion

- [ ] `documentId` généré côté backend, persisté, stable entre versions
- [ ] Objets MinIO préfixés par `{organizationId}/`
- [ ] PUT MinIO confirmé **avant** la notification à l'IA
- [ ] Timeout d'ingestion à 120 s, distinct de celui du chat
- [ ] Rejeu avec backoff sur `502`, alerte sur `500`
- [ ] Cas « PDF scanné » (`400`) traité avec un message compréhensible par l'agent
- [ ] Sur timeout : interroger `GET .../documents` avant de rejouer
- [ ] `DELETE /admin/organizations/{id}` câblé sur la résiliation d'un client

### Isolation — à vérifier explicitement en recette

- [ ] Deux organisations créées, avec des documents distincts
- [ ] Une question d'A ne cite **jamais** un document de B (vérifier via `sources`)
- [ ] `GET .../documents` de A ne liste que les documents de A
- [ ] `clear` sur A laisse B intacte
- [ ] Un `organization_id` inconnu répond « information indisponible »,
      **sans se rabattre sur les documents d'une autre structure**
- [ ] Un `organization_id` mal formé est rejeté en `400`

---

## 12. Limites connues et évolutions

### Limites structurelles — à connaître, elles ne seront pas corrigées à court terme

**La mémoire conversationnelle vit en RAM, dans le process.** Un redémarrage
l'efface, et **plusieurs instances du service IA ne la partagent pas**. Si vous
placez l'IA derrière plusieurs répliques ou un autoscaler, les questions de suite
casseront de façon aléatoire selon l'instance atteinte. Deux options : une
affinité de session sur le `session_id`, ou une instance unique. À trancher
ensemble avant toute mise à l'échelle.

**Les endpoints `/session/*` ne sont pas scopés par organisation.** Qui connaît un
`session_id` lit la conversation. Générez des UUID v4 et ne les exposez pas.

**Les fichiers audio ne sont jamais purgés.** Une réponse vocale produit N+1
fichiers (les morceaux, plus l'assemblage) dans `static/`. Sans ménage, le disque
se remplit. Définissez une politique de rétention avec l'exploitation — quelques
heures suffisent, l'usager ne réécoute pas.

**Aucune authentification sur le service IA.** Toute la sécurité repose sur
l'isolation réseau décrite en §4.

### Évolutions envisagées

| Sujet | Intérêt | Statut |
|---|---|---|
| Ingestion asynchrone (`202` + suivi) | gros documents sans timeout | à la demande |
| Secret partagé backend ↔ IA | défense en profondeur | ~10 lignes, à arbitrer |
| Mémoire externalisée (Redis) | mise à l'échelle horizontale | non planifié |
| Purge automatique des audio | exploitation | non planifié |
| OCR sur PDF scannés | cas fréquent en production | à évaluer |

---

## Contacts et références

| Document | Contenu |
|---|---|
| `API.md` | contrat brut de l'API, tous endpoints |
| `README.md` | fonctionnement interne du pipeline, latences mesurées, journalisation |
| `INTEGRATION_BACKEND.md` | ce document — contrat d'intégration backend |

**Équipe IA** : Awa FAYE, Fallilou Mbacké GUEYE
**Équipe Backend** : Mouhamadou Lamine Ndiaye

Toute question sur un écart entre ce document et le comportement observé est un
bug de l'un ou de l'autre : signalez-la avec le `session_id` et l'horodatage.
