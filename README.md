# TontumaBot V3

Assistant administratif multilingue (Wolof / Français) avec RAG, STT, TTS et API REST + SSE.

> **Vous reprenez le projet ?** Trois lectures dans cet ordre :
> [Démarrage rapide](#démarrage-rapide) pour le lancer,
> [Latence mesurée](#latence-mesurée) pour comprendre
> où passe le temps et pourquoi le code est écrit ainsi, et
> [Pièges connus et état du chantier](#pièges-connus-et-état-du-chantier) pour
> savoir ce qui est vérifié, ce qui ne l'est pas, et ce qui a déjà coûté du temps.
>
> Le principe qui gouverne la plupart des décisions : **la synthèse vocale coûte
> environ deux secondes par seconde d'audio produite.** Tout ce qui rallonge la
> réponse rallonge l'attente devant la borne, et les commentaires du code citent
> la mesure qui justifie chaque choix plutôt que de l'affirmer.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TontumaBot V3 (FastAPI)                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │   STT    │→ │  Detect  │→ │Translate │→ │  Intent  │→ │   RAG    │       │
│  │ wo:HuBERT│  │  Langue  │  │ WO → FR  │  │  Router  │  │ (Hybrid  │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │  + MMR   │       │
│                                                          │  + Rerank)       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  └──────────┘       │
│  │   TTS    │← │Translate │← │   LLM    │← │ Context  │                     │
│  │Oolel-Voices│ │ FR → WO │  │ (Groq/   │  │ + Prompt │                     │
│  └──────────┘  └──────────┘  │  Gemini/ │  └──────────┘                     │
│                              │  Local)  │                                   │
│                              └──────────┘                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Stack technique

| Composant | Technologie |
|-----------|-------------|
| API       | FastAPI + Uvicorn (REST pour l'envoi, SSE pour la réponse) |
| LLM       | Groq (openai/gpt-oss-120b, **raisonne avant de répondre**), Gemini, Local (Qwen2.5-7B 4bit) |
| Embeddings| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 |
| Vector DB | ChromaDB (persistant) |
| Retrieval | Hybrid BM25 + Vectoriel → MMR → Cross-encoder reranker (index + embeddings cachés) |
| Traduction| NLLB-200-distilled-600M (WO↔FR) |
| STT       | wo : Wolof-HuBERT-CTC · fr : Whisper       |
| TTS       | **Oolel-Voices** (voice cloning, soynade-research/Oolel-Voices) |
| Frontend  | HTML/JS vanilla (REST + SSE) |

## Démarrage rapide

```bash
./demarrer.sh
```

Le script fait tout, sur une machine neuve comme sur une machine déjà équipée :
il cherche un Python ≥ 3.10, crée l'environnement virtuel (ou **adopte celui qui
existe** sans rien réinstaller), choisit la roue PyTorch adaptée à la machine,
vérifie `.env` et la clé LLM, annonce le backend de calcul détecté, télécharge
les poids manquants, compte les fragments indexés, puis démarre le serveur.

| Option | Effet |
|--------|-------|
| `--no-start` | tout préparer sans lancer le serveur |
| `--skip-models` | ne pas télécharger les poids (déjà en cache) |
| `--reinstall` | repartir d'un environnement virtuel vierge |

À la main, si besoin :

```bash
python -m venv venv && venv/bin/pip install -r requirements.txt
cp .env.example .env          # puis renseigner GROQ_API_KEY
venv/bin/python app.py        # → http://localhost:8008
```

> **Python 3.10 minimum** : le code annote les types avec la syntaxe `str | None`,
> évaluée à l'exécution et rejetée par 3.9.

> **Warm-up au démarrage** (~67 s sur GPU, plusieurs minutes sur CPU) : les
> modèles sont préchargés, **et le TTS fait une synthèse à vide**. Les noyaux de
> calcul se compilent au premier passage et cette compilation se paie une fois
> par processus — mesuré, la première synthèse coûte 1,5 fois les suivantes.
> Autant la payer au lancement, où personne n'attend devant la borne.
> Désactivable via `WARMUP_ON_START=false` / `WARMUP_TTS=false`.

---

## Backend de calcul et mémoire GPU

`src/device.py` donne une réponse unique et surchargeable au choix du backend
(`cuda` > `mps` > `cpu`), pour tous les composants. Avant lui, chacun décidait
seul et les résultats divergeaient : NLLB et le TTS basculaient sur MPS, le STT
restait sur CPU, le reranker laissait sentence-transformers trancher.

```bash
cd src && python device.py      # rapport : backends disponibles, choix par composant
```

`DEVICE=cpu` impose le backend à tout le pipeline ; `<COMPOSANT>_DEVICE`
(`STT_DEVICE`, `NLLB_DEVICE`, `TTS_DEVICE`…) isole un seul modèle. Un backend
demandé mais indisponible ne fait pas échouer le démarrage : repli sur le
meilleur disponible, en le signalant. `/health` renvoie l'état complet sous
`device`.

### Le plafond MPS — à ne pas retirer sur Mac

Sur Apple Silicon, `MPS_MEMORY_RATIO` (0.9 par défaut) borne la mémoire que
PyTorch réserve au pilote Metal. **Sans ce plafond, le serveur se fait tuer par
le système.** Mesuré sur M1 16 Go :

| | mémoire réservée au pilote MPS |
|---|---|
| modèles chargés | 6,85 Go |
| **après une seule synthèse, sans plafond** | **16,71 Go** |
| après une synthèse, plafond 0.9 | 9,80 Go |
| après trois synthèses, plafond 0.9 | 9,82 Go (stable) |

Les tenseurs réellement vivants ne pèsent que 6,5 Go : l'écart est du cache
d'allocateur. Le décodeur auto-régressif du TTS agrandit son cache KV d'un cran
à chaque pas, réclame donc une taille de bloc inédite à chaque pas, et
l'allocateur en conserve une par taille. PyTorch l'y autorise jusqu'à 1,7 fois
la mémoire de travail recommandée par Metal. Le plafond ne coûte rien en
latence (42-47 s contre 49 s sur la même réponse).

Le symptôme, si quelqu'un le retire : `zsh: killed python app.py`, sans autre
message. La mémoire ne part pas dans le processus — son RSS reste à ~1,1 Go —
mais dans le pilote graphique, invisible avec un simple `ps`.

---

## Latence mesurée

La synthèse vocale domine tout le reste. Sur une réponse type, la chaîne RAG
(récupération, LLM, traduction) prend ~11 s quand le TTS en prend 30 à 50.

| Poste | Coût |
|-------|------|
| Synthèse (MPS) | ~2 s par seconde d'audio produite (RTF 1,96-2,40) |
| Synthèse (CPU) | RTF 5,69-6,78 — **trois fois plus lent** |
| Chaîne RAG complète | ~11 s |
| Traduction NLLB (une réponse) | ~1 à 2,5 s |
| LLM (Groq) | ~0,8 s |

### Les quatre leviers appliqués

**1. Attention optimisée rétablie sur le décodeur T3** (`_accelerer_attention`
dans `src/tts_Ooleil/tts.py`). Sa boucle de génération réclamait
`output_attentions=True` à chaque pas. Ces poids ne servent qu'à
l'`AlignmentStreamAnalyzer`, que le modèle ne construit que pour les
checkpoints multilingues — et celui-ci ne l'est pas
(`text_tokens_dict_size=704`). Les matrices étaient calculées puis jetées, mais
leur seule demande faisait retomber Llama de SDPA sur l'attention manuelle.
Mesuré à graine fixe, **18,6 s → 10,8 s** sur une phrase de 85 caractères, pour
un audio identique. La fonction se désactive d'elle-même si le checkpoint est
multilingue.

**2. Identité vocale préparée une seule fois.** `generate(audio_prompt_path=…)`
relançait le décodage du fichier de référence (36 s d'audio) et l'encodeur de
voix à **chaque appel**. Elle est désormais extraite au chargement.

**3. Découpage de la synthèse en morceaux** (`TTS_CHUNK_CHARS`, 200 par défaut).
Le gain n'est pas le débit — une fois l'attention corrigée, le coût est
proportionnel à la durée d'audio — mais **le temps jusqu'au premier son** et une
allocation mémoire bornée par appel. Sur une réponse de 903 caractères :
premier morceau prêt à 23 s au lieu de 123 s pour l'énoncé complet.

**4. Réponses plus courtes** (`LLM_MAX_PHRASES`). La consigne système explique
au modèle *pourquoi* : sa réponse est lue à voix haute. Mesuré sur les 20
questions de référence :

| | avant | après |
|---|---|---|
| longueur moyenne (wolof) | 332 car. | **222 car.** |
| longueur maximale | 903 car. | **473 car.** |
| réponses > 400 car. | 8/20 | **1/20** |
| synthèse estimée, moyenne | 47 s | **31 s** |
| synthèse estimée, au pire | 126 s | **66 s** |

La consigne interdit aussi de comprimer en devenant vague : mesuré, sommé
d'être bref, le modèle remplaçait « 48 à 72 heures » par « le délai varie selon
l'urgence ». Une réponse courte et floue est pire qu'une réponse longue.

> **Piège `LLM_MAX_TOKENS`** : `gpt-oss-120b` **raisonne avant de répondre**, et
> sa trace interne (~200 jetons, 750-840 caractères) est comptée dans
> `max_tokens`. À 220 jetons, tout le budget y passait et **7 réponses sur 20
> revenaient vides**. Le plafond est à 800 : c'est un garde-fou contre une
> génération qui s'emballe, pas l'outil de mise en forme. Une réponse vide est
> désormais signalée et remplacée par le passage de contexte le plus pertinent,
> au lieu de traverser le pipeline en silence.

---

## Docker

> **Sur Mac Apple Silicon, ne pas utiliser Docker pour une borne en service.**
> Un conteneur n'a pas accès à Metal : Docker Desktop fait tourner une VM Linux
> où MPS n'existe pas, donc tout passe sur CPU. Mesuré, la synthèse passe d'un
> facteur 2,0 à 6,8 fois la durée de l'audio — une réponse parlée de quinze
> secondes demande 100 s au lieu de 30 s. Utiliser `./demarrer.sh`.
> Le conteneur sert à livrer sur un serveur Linux et à disposer d'un
> environnement reproductible.

```bash
./setup.sh                                  # tout, de bout en bout
# ou à la main :
cp .env.example .env                        # renseigner GROQ_API_KEY
docker compose build
docker compose run --rm download-models     # ~4 Go, une seule fois
docker compose up -d                        # → http://localhost:8008/borne
```

Trois points à connaître avant de construire :

- **Mémoire de la VM** : les modèles pèsent 6,5 Go. Docker Desktop n'alloue par
  défaut qu'une fraction de la RAM de l'hôte ; en dessous de 10 Go le conteneur
  est tué sans message explicite (Réglages → Resources → Memory).
- **Sur ARM**, l'installation de PyTorch bascule sur PyPI. L'index CPU de
  PyTorch n'y publie `torchaudio` que jusqu'à la version 2.0.2, incompatible
  avec un `torch` récent : la résolution échouerait. Le `Dockerfile` branche sur
  `TARGETARCH`.
- **`static/` est monté depuis l'hôte.** Les audios TTS s'y écrivent, et depuis
  le découpage une réponse produit N+1 fichiers (les morceaux diffusés, plus
  l'assemblage) de ~500 Ko. Sans ce montage ils gonflent la couche du conteneur
  et disparaissent au premier `--force-recreate`.

### Ce qui ne voyage pas avec le dépôt

Trois chemins sont montés par `docker-compose.yml` mais **absents de git**. Un
montage bind sur un dossier inexistant ne produit aucune erreur : il crée un
dossier vide.

| Chemin | Taille | Conséquence si absent |
|--------|--------|------------------------|
| `data/chroma/` | 2,2 Mo | **la borne démarre, `/health` dit `ok`, et elle ne sait répondre à rien** |
| `src/stt_wolof-hubert-ctc/` | 361 Mo | le STT wolof retombe sur l'identifiant Hub et se retéléchargera |
| `uploads/*.pdf` | 2 fichiers | les sources qui permettraient de réindexer |

---

## Endpoints REST

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET`   | `/` · `/borne` · `/borne/simple` · `/admin` | Pages servies |
| `GET`   | `/health` | État du service, backends et mémoire |
| `POST`  | `/ask` | Question texte → réponse **SSE** |
| `POST`  | `/ask/audio` | Audio → STT → pipeline → réponse **SSE** |
| `GET`   | `/session/{id}` | État de la mémoire conversationnelle |
| `POST`  | `/session/{id}/reset` | Vider l'historique, garder la session |
| `DELETE`| `/session/{id}` | Oublier la session |
| `GET`   | `/static/{fichier}.wav` | Audio produit (via `audio_url`) |
| `POST`  | `/admin/documents` | Ingérer un document (TXT/MD/PDF) |
| `GET`   | `/admin/documents` | Lister les documents indexés |
| `DELETE`| `/admin/documents/{id}` | Supprimer un document |
| `POST`  | `/admin/documents/clear` | **Tout effacer** |
| `GET`   | `/translate` | Test de traduction NLLB, sans RAG |
| `POST`  | `/eval/ragas` | Évaluation sur un jeu de test fourni |

> **Aucune authentification, CORS ouvert à `*`** — `/admin/*` compris. Quiconque
> joint le service peut effacer la base documentaire. Le fichier `api.key` du
> dépôt n'est lu par aucune route. En exposition réseau, protégez `/admin/*` en
> amont ou n'exposez que `localhost`.

Contrat complet dans **[`API.md`](API.md)**.

---

## API SSE

`POST /ask` et `POST /ask/audio` répondent en `text/event-stream`. Le contrat
complet — événements, charges utiles, paramètres, clients d'exemple en
JavaScript et en Java, codes d'erreur — est dans **[`API.md`](API.md)**.

Il n'est pas repris ici : deux descriptions du même contrat finissent toujours
par diverger, et c'est déjà arrivé sur ce projet.

Ce qu'il faut en retenir côté conception :

- **L'étape `tts` porte le texte de la réponse.** Le `result` n'arrive qu'une
  fois l'audio produit, jusqu'à deux minutes plus tard sur une réponse longue —
  sans cet événement, la borne parlerait avant d'afficher ce qu'elle dit.
- **`tts_chunk` diffuse la synthèse morceau par morceau.** Le premier son est
  disponible à 37 s là où l'énoncé complet demande 153 s. Les clients qui les
  ignorent attendent simplement `audio_url` dans `result`.
- **`alerte_langue` et `alerte_nombres` signalent sans corriger** : dans les deux
  cas le mal est déjà fait en amont, et le pipeline préfère le dire que de le
  taire.

```bash
curl -N -X POST http://localhost:8008/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Comment faire un passeport ?","tts":false}'
```

---

## Interfaces

Deux pages, servies par le même backend, avec des besoins différents.

| | `static/borne.html` (+ `borne.js`, `borne.css`) | `static/index.html` |
|---|---|---|
| Usage | la borne en service, tactile, avec micro | démo et mise au point |
| Lecture audio | **enchaîne les morceaux** dès le premier | `<audio controls>` sur le fichier complet |
| TTS demandé | sur les questions vocales uniquement | case à cocher, activée par défaut |
| Langue de la dictée | deux boutons « appuyer pour parler » | sélecteur « Je parle… » |
| Jauge de progression | trait sur la barre d'état | dans la bulle de traitement |
| Trace du pipeline | non | oui, détaillée sous la réponse |

`borne-simple.html` partage `borne.js` et `borne.css` : toute modification du
comportement vaut pour les deux bornes. La jauge est créée depuis le JavaScript
et se greffe sur `.statusbar`, pour n'avoir ni HTML dupliqué ni page à oublier.

### La jauge de progression

Chaque étape du pipeline porte un jalon — la place estimée de sa fin sur la
course. Les poids ne sont pas uniformes : quand le TTS est demandé, la synthèse
pèse à elle seule plus que toute la chaîne RAG, et des parts égales donneraient
une jauge qui bondit à 90 % puis paraît figée.

L'aiguille ne saute pas d'un jalon au suivant : elle s'en approche par fractions,
vite puis de plus en plus lentement. Une étape de vingt secondes garde ainsi une
barre qui avance, **sans jamais dépasser le jalon annoncé par le serveur**.

L'échelle s'adapte au travail demandé, connu en cours de route : le TTS est
annoncé par `start`, la langue par `detect`. Sans synthèse, les jalons de la
chaîne RAG s'étirent sur toute la course — sinon la réponse arriverait sur une
barre à 44 %. Vérifié en rejouant les séquences d'étapes réelles :

| Scénario | Fin de course |
|---|---|
| vocale + TTS (1 ou N morceaux) | 97 % |
| clavier français, sans TTS | 95 % |
| clavier wolof, sans TTS | 95 % |

La jauge ne recule jamais, et les étapes inconnues ou absentes sont ignorées
sans la faire redescendre — `alerte_nombres` en est une, réellement émise.

### La lecture enchaînée (borne uniquement)

La synthèse produit environ deux fois moins vite qu'on n'écoute : la file de
morceaux se vide donc en cours de route. La borne reste alors en état
`SPEAKING`, **silencieuse, sans rendre la main** — repasser au repos couperait
l'usager en plein milieu d'une phrase. `STOP` vide la file et abandonne la
lecture ; les morceaux encore en vol ne la relancent pas.

---

## Choix de la langue d'entrée

**Sur le chemin vocal, la langue n'est pas détectée : elle est déclarée.** Elle
sélectionne le moteur STT *avant* qu'il existe une transcription, et les deux
moteurs sont des architectures différentes (Wolof-HuBERT-CTC en décodage CTC,
Whisper pour le français). Il n'y a rien à analyser tant qu'on n'a pas choisi.

| Chemin | Origine de la langue | `trace.input_lang_source` |
|--------|----------------------|---------------------------|
| Borne, micro | bouton 🇸🇳 WOLOF / 🇫🇷 FRANÇAIS (`borne.js`, `pressLang`) | `indice` |
| Chat, micro | sélecteur « Je parle… » à côté du bouton micro | `indice` |
| Chat ou API, texte | **`detect_language()` sur le texte** | `détection` |

La langue déclarée arrive en `lang_hint` et **l'emporte sur la détection**
(`src/pipeline.py`, étape 1) : sur des phrases courtes, avec des emprunts et du
code-switching, une détection ne peut que se tromper là où l'usager, lui, sait
ce qu'il parle.

Cette langue gouverne ensuite quatre décisions : la traduction WO→FR de la
question, le choix du prompt système, la traduction FR→WO de la réponse, et la
préparation du texte pour la synthèse.

### Le contrôle a posteriori

Se tromper de bouton n'est rattrapable par rien : la transcription est déjà
faite par le mauvais moteur — HuBERT-CTC sur du français rend de la bouillie
phonétique, Whisper forcé en français sur du wolof invente des mots — et la
réponse repart dans la mauvaise langue.

`contredit()` (`src/language/detector.py`) ne corrige donc pas, il **rend
l'erreur visible** : la détection tourne malgré tout sur la transcription et,
si elle contredit nettement la langue annoncée, le pipeline émet
`alerte_langue` et les deux interfaces l'affichent dans le fil.

Le seuil vit dans `detector.py` parce que c'est ce module qui sait que
`detect_language` retombe sur `fr` quand rien ne tranche : sans lui, « Waaw. »
déclarée en wolof lèverait une alerte à chaque fois. Le gagnant doit marquer au
moins 6 points **et** doubler le perdant — un mot wolof reconnu en vaut 3, un
mot français 2. Vérifié :

| Cas | Verdict |
|-----|---------|
| français franc déclaré wolof | alerte (`wo=0`, `fr=18`) |
| wolof franc déclaré français | alerte (`wo=29`, `fr=2`) |
| langues concordantes | silence |
| « Waaw. », transcription vide | silence |
| « carte d'identité ci » (mélangé) | silence |

> **Piège corrigé** : le chat n'envoyait aucune langue pour la dictée, et
> `/ask/audio` retombait sur son défaut `Form("wo")`. Une question posée en
> français au micro était donc transcrite par le modèle wolof, traduite WO→FR,
> puis **répondue en wolof** — alors que la page affiche « Détecteur actif ».
> Le sélecteur a été ajouté pour ça.

---

## Mémoire conversationnelle

`src/memory.py` tient un tableau de messages par `session_id`, en français — la
langue pivot du pipeline. Il sert à deux choses : comprendre les questions de
suite (« et combien ça coûte ? » n'a de sens qu'avec le tour précédent) et
compléter la requête de recherche quand la question est trop courte
(`_contextualize` / `_sujet_courant` dans `src/pipeline.py`, une heuristique qui
préfixe le sujet courant — **pas** un appel LLM de reformulation).

Au-delà de `MEMORY_MAX_MESSAGES` (10 entrées = 5 échanges), la mémoire est
**vidée** — la session, elle, continue. C'est une remise à zéro, pas une
fermeture : aucun code d'erreur, aucune session à recréer.

**Cette remise à zéro doit rester visible à l'écran.** Sans elle, les questions
de suite cessent d'un coup d'être comprises et l'usager n'a aucune explication.
Les deux interfaces affichent donc un compteur `🧠 n / 10` et un message dans le
fil. L'avis se place **avant** la bulle de réponse : sur le chemin vocal, celle-ci
existe déjà quand l'état mémoire arrive, et un simple `appendChild` le
rejetterait sous la réponse, où il se lirait comme un commentaire de celle-ci.

L'historique n'est **jamais** une source de faits : seule la base documentaire
fonde une réponse. C'est écrit dans les deux prompts système.

Endpoints associés : `GET /session/{id}`, `POST /session/{id}/reset`,
`DELETE /session/{id}`. La mémoire vit en RAM, par processus — un redémarrage
l'efface, et un déploiement multi-borne demanderait un magasin partagé.

---

## Variables d'environnement (.env)

**`.env.example` fait référence** : il porte les 30 réglages avec, pour chacun,
la mesure ou la raison qui justifie sa valeur. Recopier cette liste ici la
condamnerait à diverger — elle avait déjà pris du retard. Ne sont rappelés
ci-dessous que ceux dont un mauvais réglage casse quelque chose.

| Réglage | Défaut | Pourquoi y faire attention |
|---------|--------|----------------------------|
| `MPS_MEMORY_RATIO` | `0.9` | Sur Mac, le retirer fait tuer le serveur par le système. `0` = pas de plafond |
| `LLM_MAX_TOKENS` | `800` | **Ne pas serrer** : la trace de raisonnement du modèle y est comptée. À 220, 7 réponses sur 20 revenaient vides |
| `LLM_MAX_PHRASES` | `3` | Règle la longueur, donc la durée de synthèse. C'est la consigne système, pas le plafond de jetons, qui fait le travail |
| `TTS_CHUNK_CHARS` | `200` | Longueur d'un morceau. `0` désactive le découpage — et avec lui le premier son anticipé |
| `TTS_N_STEPS` | `10` | Itérations du vocodeur. `4` ≈ 2,5× plus rapide, compromis de qualité jamais évalué à l'oreille |
| `WARMUP_TTS` | `true` | Déclenche une synthèse à vide (~60 s). En chargement paresseux, ce coût tomberait dans la première requête |
| `MEMORY_MAX_MESSAGES` | `10` | 10 entrées = 5 échanges, puis la mémoire est vidée |
| `DEVICE` | `auto` | `cuda` > `mps` > `cpu`. Surchargeable par composant (`TTS_DEVICE`, `NLLB_DEVICE`…) |
| `LOG_LEVEL` | `INFO` | `DEBUG` trace chaque étape du pipeline — le premier réflexe quand une requête se fige |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Modèle **à raisonnement** : sa trace interne compte dans `LLM_MAX_TOKENS` |

---

## Modèles utilisés

| Tâche | Modèle | Source |
|-------|--------|--------|
| LLM (cloud) | openai/gpt-oss-120b | Groq |
| LLM (cloud) | gemini-2.0-flash | Google |
| LLM (local) | Qwen2.5-7B-Instruct (4bit) | HuggingFace |
| Embeddings | paraphrase-multilingual-MiniLM-L12-v2 | SBERT |
| Traduction WO↔FR | NLLB-200-distilled-600M | bilalfaye (HF) |
| STT Wolof | Wolof-HuBERT-CTC | soynade-research (local ou Hub) |
| STT Français | Whisper (small par défaut) | openai (cache HF) |
| TTS (principal) | **Oolel-Voices** | soynade-research (HF) |
| Reranker | ms-marco-MiniLM-L-6-v2 | cross-encoder |

---

## Structure du projet

```
TontumaBot_RAG1_V3/
├── demarrer.sh                # Démarrage natif (venv, torch, modèles, serveur)
├── setup.sh                   # Installation Docker de bout en bout
├── app.py                     # FastAPI (REST + SSE) + warm-up des modèles
├── download_models.py         # Pré-téléchargement des modèles HF
├── pipeline_rag.py            # Script pipeline autonome (hors serveur)
├── questions_test_wo.json     # 20 questions wolof de référence
├── resultat.py                # Résultats d'un passage sur ces 20 questions
├── requirements.txt
├── .env                       # Variables d'environnement
├── 8_1_c.wav                  # Voix de référence du clonage vocal (36 s)
├── static/
│   ├── borne.html             # Borne tactile (production)
│   ├── borne-simple.html      # Variante allégée, même JS
│   ├── borne.js / borne.css   # Machine à états, jauge, lecture enchaînée
│   ├── index.html             # Démo web (trace du pipeline visible)
│   ├── admin.html             # Interface d'administration RAG
│   └── response_*.wav         # Audios produits — jamais purgés (cf. Pièges)
├── src/
│   ├── config.py              # Configuration (.env) + plafond mémoire MPS
│   ├── device.py              # Choix du backend CPU / CUDA / MPS
│   ├── journal.py             # Journalisation : niveaux, identifiant de requête
│   ├── pipeline.py            # Pipeline RAG complet (+ callback SSE)
│   ├── memory.py              # Mémoire conversationnelle par session
│   ├── vectorstore.py         # ChromaDB + BM25/embeddings cachés + MMR
│   ├── ingestion.py           # Indexation documents (TXT/MD/PDF)
│   ├── input/stt.py           # STT bilingue (HuBERT wolof / Whisper français)
│   ├── language/detector.py   # Détection de langue + contrôle d'une langue déclarée
│   ├── translation/
│   │   ├── nllb.py            # Traduction WO↔FR, phrase par phrase en lot
│   │   ├── nombres.py         # Nombres : chiffres ↔ lettres, réalignement FR/WO
│   │   └── prononciation.py   # Préparation du texte pour la synthèse
│   ├── intent/router.py       # Router d'intention
│   ├── retrieval/
│   │   ├── hybrid.py          # BM25 + vectoriel
│   │   ├── filtered.py        # Recherche filtrée (orientation)
│   │   └── reranker.py        # Cross-encoder
│   ├── generation/llm.py      # LLM (Groq / Gemini / local) + consigne de brièveté
│   ├── tts_Ooleil/tts.py      # TTS Oolel-Voices (découpage, diffusion, préchauffage)
│   └── evaluation/ragas_eval.py  # Évaluation RAGAS
└── data/                      # Documents seed + ChromaDB persistant
```

### Les trois modules de traduction

`nombres.py` et `prononciation.py` existent pour une raison mesurée : NLLB
altère les nombres. « cinq mille francs » ressort en wolof par « junni »
(mille), alors que « 5000 francs » traverse intact — d'où l'ordre donné au LLM
d'écrire les nombres en chiffres. Sur 20 questions réelles, 8 réponses portaient
un faux nombre, dont « composez le 3333 » rendu par « 333 », le numéro d'urgence
de l'hôpital. Le français fait autorité : les nombres y sont réinjectés position
par position quand les deux textes en portent autant, et l'écart est signalé
sinon (`alerte_nombres`).

`prononciation.py` fait le chemin inverse **juste avant la synthèse** :
Oolel-Voices ne lit correctement que du wolof en toutes lettres, et certains
motifs (« APIX », « 8:00 ») détruisent l'énoncé entier. L'affichage, lui, garde
ses chiffres.

---

## Journalisation et débogage

Le projet traçait avec `print()` : pas d'horodatage, pas de niveau, aucun moyen
de baisser le volume, et surtout **aucun lien entre une ligne et la requête qui
l'a produite**. Sur une borne, FastAPI exécute le pipeline dans un fil séparé et
deux usagers simultanés entrelacent leurs traces sans recours.

`src/journal.py` remplace tout ça. Une ligne ressemble à ceci :

```
11:39:35.844  INFO    5582  pipeline     procedure · 448 car. · total=10127 ms   wo→fr=1099  intent=49  recher=102  fr→wo=8824
└ heure à la ms       └ niveau  └ requête   └ composant
```

L'horloge va à la milliseconde parce que tout ce qu'on débogue ici est une
affaire de latence. L'identifiant de requête tient en quatre caractères, assez
pour distinguer sans manger la ligne.

### Ce qu'on lit en pratique

**La ligne de fin de requête donne la répartition de la latence** — c'est elle
qu'on regarde quand « la borne est lente ». Sur l'exemple ci-dessus, la
traduction FR→WO pèse 8,8 s sur 10,1 s : le poste fautif se désigne sans avoir
à ouvrir la trace JSON.

**En `DEBUG`, chaque étape est horodatée** : on voit où une requête se fige.

```
11:40:05.785  DEBUG   6b8f  pipeline     → detect lang=wo source=indice
11:40:05.785  DEBUG   6b8f  pipeline     → translate_in
11:40:29.498  DEBUG   6b8f  pipeline     → intent intent=procedure
```

### Réglages

| | Défaut | Effet |
|---|---|---|
| `LOG_LEVEL` | `INFO` | `DEBUG` ajoute la progression étape par étape |
| `LOG_FILE` | vide | chemin d'un fichier — rotation à 5 Mo, 3 archives |
| `LOG_COULEUR` | `auto` | couleurs si un humain regarde, jamais dans un fichier |

### Points de conception

**Le contexte de requête ne traverse pas les fils d'exécution tout seul.**
`run_in_executor` n'emporte pas les `contextvars` : `app.py` rouvre donc
explicitement le contexte à l'intérieur du fil. Sans ce détail, tout ce que le
pipeline journalise perdrait son identifiant — c'est-à-dire l'essentiel.

**Les bibliothèques bavardes sont muselées** (`transformers`, `datasets`,
`httpx`, `urllib3`…) et les avertissements Python passent par le journal au lieu
d'écrire sur stderr. Leurs messages de dépréciation noyaient les traces utiles
au point qu'il fallait les filtrer au `grep` pour lire quoi que ce soit.

**Uvicorn est configuré avec `log_config=None`** pour que ses propres lignes
adoptent le même format : une seule trace à lire, pas deux.

**Une ligne de journal tient sur une ligne.** Les extraits de texte sont aplatis
avant d'y être insérés : un message multi-ligne casse un `grep`.

Les scripts de `benchmark/` et `src/evaluation/` gardent leurs `print()` — leur
sortie *est* le livrable, pas une trace de service.

---

## Pièges connus et état du chantier

Cette section est la mémoire du projet : ce qui a été vérifié, ce qui ne l'a pas
été, et les pièges qui ont déjà coûté du temps.

### Vérifié et mesuré

- Le plafond MPS : serveur survivant à quatre questions vocales enchaînées, mémoire
  disponible revenant à son niveau de départ après chaque requête.
- La diffusion en morceaux : chronologie relevée de bout en bout sur le serveur.
- La jauge de progression : logique rejouée sur les séquences d'étapes réelles.
- Les longueurs de réponse : 20 questions avant/après.

### Non vérifié

- **Le rendu des interfaces dans un navigateur.** La jauge, la lecture enchaînée
  et l'affichage de la mémoire sont validés par le code, des tests de logique et
  le flux SSE côté serveur — mais aucun œil humain ni automate n'a ouvert la page.
- **La construction de l'image Docker** sur ARM. Les corrections reposent sur les
  index de paquets interrogés, pas sur un `docker compose build` réussi.
- **Le test d'endurance.** La mémoire tient sur quelques requêtes ; une borne
  tourne des heures.

### Pièges

- **`static/` n'est jamais purgé.** Une réponse produit N+1 fichiers de ~500 Ko
  depuis le découpage. Rien ne les efface : une borne en service remplit son
  disque, et la panne arrivera avant que quiconque le remarque.
- **Le corpus ne voyage pas avec le dépôt** (cf. Docker). Symptôme : la borne
  démarre, `/health` répond `ok`, et elle ne sait répondre à rien.
- **`resultat.py` est daté.** Il porte les mesures d'avant la consigne de
  brièveté ; toute comparaison qui s'y réfère part d'une base périmée.
- **Ne pas lancer l'application en natif ET en conteneur en même temps** : deux
  écrivains sur le même SQLite ChromaDB.
- **`TTS_N_STEPS=4`** est un compromis de qualité audio (le vocodeur fait 4
  itérations de diffusion au lieu de 10) qui n'a jamais été comparé à l'oreille.

### Jamais mesuré

La latence du STT, alors que le chemin vocal commence par elle. Tout le travail
d'optimisation a porté sur la fin de la chaîne.

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