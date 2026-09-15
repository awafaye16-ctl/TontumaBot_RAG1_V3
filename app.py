"""TontumaBot V3 — FastAPI."""
import os
import sys

# ── Désactiver TF/Keras AVANT tout autre import ───────────────────────────
# sentence_transformers → keras → tensorflow (~20s) si on ne bloque pas ici
os.environ["USE_TF"]                = "0"
os.environ["USE_TORCH"]             = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TOKENIZERS_PARALLELISM"]= "false"
# ─────────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
sys.path.insert(0, os.path.join(BASE_DIR, "data"))

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import settings
import device
from journal import journal, configurer as configurer_journal, contexte_requete, nouvel_identifiant
from pipeline import answer as pipeline_answer
from seed_docs import get_documents
import memory
import vectorstore
import ingestion
import storage


# =============================================================================
#  Warm-up : précharge les modèles au démarrage pour éviter la latence de
#  chargement sur la première requête (embedder, BM25, reranker, NLLB, STT, TTS)
# =============================================================================

def _warmup() -> None:
    # Le backend est annoncé une fois pour toutes : c'est la première chose à
    # vérifier quand la latence dérape (un repli silencieux sur CPU multiplie
    # les temps de traduction par trois).
    log = journal("warmup")
    infos = device.infos()
    log.info("torch %s — cuda=%s mps=%s → auto=%s (DEVICE=%s)",
             infos["torch"], infos["cuda"], infos["mps"], infos["auto"], infos["override"])

    def _step(name: str, fn):
        t0 = time.perf_counter()
        try:
            fn()
            log.info("%-12s prêt en %.0f ms", name, (time.perf_counter() - t0) * 1000)
        except Exception as e:
            # Un modèle absent au démarrage n'arrête pas le service : il sera
            # rechargé à la demande, ou le composant restera indisponible.
            log.warning("%-12s ignoré : %s", name, e)

    # Embedder — partagé par toutes les organisations
    _step("embedder", vectorstore.get_embedder)

    # Corpus des organisations déjà présentes sur le disque. Le premier appel
    # d'une organisation paie sinon l'ouverture de sa base, le chargement de ses
    # vecteurs et la construction de son index BM25 — une à trois secondes,
    # visibles sur une borne tactile.
    def _corpus():
        orgs = vectorstore.organisations()[: settings.MAX_ORGANISATIONS_EN_CACHE]
        total = 0
        for o in orgs:
            try:
                total += vectorstore.prechauffer(o["organization_id"])
            except Exception as e:  # noqa: BLE001
                # Une base incompatible ou corrompue ne doit pas empêcher le
                # service de démarrer pour les autres organisations.
                log.warning("organisation %s non préchauffée : %s",
                            o["organization_id"], e)
        log.info("%d organisation(s) préchauffée(s), %d fragments", len(orgs), total)
    _step("corpus/bm25", _corpus)

    # Prototypes du router d'intention (réutilisent l'embedder ci-dessus)
    def _intent():
        from intent.router import load_prototypes
        load_prototypes()
    _step("intent", _intent)

    # Reranker cross-encoder
    def _reranker():
        from retrieval.reranker import load_model
        load_model()
    _step("reranker", _reranker)

    # Traduction NLLB (WO↔FR)
    def _nllb():
        from translation.nllb import _load_model
        _load_model()
    _step("nllb", _nllb)

    # STT wolof (HuBERT-CTC) — optionnel
    if settings.WARMUP_STT:
        def _stt_wo():
            from input.stt import load_model, WOLOF
            load_model(WOLOF)
        _step("stt/wo", _stt_wo)

        # Le moteur français (Whisper) est plus lourd : préchargé seulement si
        # STT_WARMUP_FR=true, sinon chargé au premier appui sur « Français ».
        if settings.STT_WARMUP_FR:
            def _stt_fr():
                from input.stt import load_model, FRANCAIS
                load_model(FRANCAIS)
            _step("stt/fr", _stt_fr)

    # TTS (Oolel-Voices) — optionnel
    if settings.WARMUP_TTS:
        def _tts():
            # Le chargement seul ne suffit pas : la première synthèse d'un
            # processus compile les noyaux de calcul et coûte le double des
            # suivantes. Elle est faite ici, pas devant l'usager.
            from tts_Ooleil.tts import prechauffer
            prechauffer()
        _step("tts", _tts)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # La journalisation s'installe avant tout le reste : sans elle, les lignes
    # émises pendant le chargement des modèles partiraient en `print` brut et
    # échapperaient au niveau, au fichier et à l'horodatage.
    configurer_journal()
    demarrage = journal("serveur")
    if settings.WARMUP_ON_START:
        demarrage.info("préchargement des modèles au démarrage")
        t0 = time.perf_counter()
        # Chargement bloquant hors de l'event loop
        await asyncio.get_running_loop().run_in_executor(None, _warmup)
        demarrage.info("warm-up terminé en %.1f s", time.perf_counter() - t0)
    else:
        demarrage.info("warm-up désactivé — les modèles se chargeront à la demande")
    demarrage.info("prêt sur %s:%s", settings.HOST, settings.PORT)
    yield
    demarrage.info("arrêt du serveur")


app = FastAPI(title="TontumaBot V3", version="3.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR  = os.path.join(BASE_DIR, "static")
UPLOAD_DIR  = os.path.join(BASE_DIR, "uploads")
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

SEED_DOCS, SEED_FILTERED = get_documents()


# =============================================================================
#  Schémas Pydantic
# =============================================================================

class AskRequest(BaseModel):
    question:        str
    organization_id: str | None = None  # UUID — base de connaissances interrogée
    provider:        str | None = None  # groq | gemini | local
    tts:             bool       = False
    lang:            str | None = None  # 'wo' | 'fr' — sinon détection automatique
    session_id:      str | None = None  # mémoire conversationnelle de cette session


class DocumentPointeur(BaseModel):
    """Notification d'ingestion : le fichier est déjà sur le stockage objet.

    Le backend dépose sur MinIO puis envoie ce pointeur ; l'IA va lire l'objet.
    `documentId` est l'identifiant du backend : le réutiliser republie le même
    document au lieu d'en empiler une seconde version.
    """
    documentId: str
    bucket:     str
    objectKey:  str
    title:      str | None = None
    category:   str | None = None


class RagasRequest(BaseModel):
    """Jeu de test pour l'évaluation RAGAS."""
    test_cases: list[dict]   # liste de {question, reference_answer, category?, language?}
    provider:   str = "groq"


# =============================================================================
#  SSE — Server-Sent Events pour la réponse en streaming
# =============================================================================
#
#  Modèle : REST pour l'envoi (POST), SSE pour la réponse (text/event-stream).
#
#  Événements émis (chacun : "event: <nom>\ndata: <json>\n\n") :
#    status  { step: "start|stt|detect|translate_in|intent|retrieval|llm|translate_out|tts", ... }
#    status  { step: "tts_chunk", index, total, audio_url }   ← un par morceau
#    result  { response, response_fr, response_wo?, qr_code?, audio_url?, lang, trace }
#    error   { message }
#    done    {}
#
#  L'audio n'est PAS envoyé dans le flux (SSE = texte) : on renvoie une URL
#  `audio_url` (fichier servi via /static) que le client récupère en GET.
#
#  La synthèse étant le poste le plus lent (jusqu'à 123 s sur une réponse
#  longue), elle est diffusée morceau par morceau : chaque `tts_chunk` porte
#  l'URL d'un fragment lisible immédiatement, dans l'ordre. Un client qui les
#  enchaîne parle dès 23 s. Ceux qui les ignorent attendent `audio_url` dans
#  `result`, qui reste la réponse entière en un seul fichier.

SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _pipeline_sse(question: str, provider: str, tts: bool, tts_out: str,
                        lang_hint: str | None = None, session_id: str | None = None,
                        organization_id: str | None = None):
    """Exécute le pipeline (dans un thread) et streame la progression en SSE.

    Le pipeline synchrone tourne dans un executor ; ses callbacks `progress`
    sont relayés vers ce générateur async via une queue thread-safe.
    """
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()

    # Un identifiant par requête, porté par toutes les lignes de journal
    # qu'elle produira. Deux usagers qui parlent en même temps deviennent
    # démêlables — sans lui, leurs traces s'entrelacent sans recours.
    rid = nouvel_identifiant()
    log = journal("requete")

    # La mémoire est lue avant le tour et complétée après : le pipeline reste
    # sans état, l'historique appartient à la session.
    conv    = memory.get(session_id)
    history = conv.history() if conv else None

    def progress(step: str, info: dict):
        loop.call_soon_threadsafe(q.put_nowait, ("status", {"step": step, **info}))

    def run():
        # `run_in_executor` n'emporte PAS le contexte de l'appelant : sans ce
        # bloc, tout ce que le pipeline journalise depuis son fil d'exécution
        # perdrait l'identifiant de la requête, c'est-à-dire l'essentiel.
        with contexte_requete(rid):
            debut = time.perf_counter()
            log.info("question (%d car.) · org=%s · provider=%s · tts=%s · langue=%s",
                     len(question), organization_id or "-", provider, tts,
                     lang_hint or "à détecter")
            try:
                result = pipeline_answer(
                    question,
                    organization_id = organization_id,
                    provider      = provider,
                    tts           = tts,
                    tts_out       = tts_out,
                    lang_hint     = lang_hint,
                    seed_docs     = SEED_DOCS,
                    seed_filtered = SEED_FILTERED,
                    progress      = progress,
                    history       = history,
                )
                log.info("terminée en %.1f s", time.perf_counter() - debut)
                loop.call_soon_threadsafe(q.put_nowait, ("result", result))
            except Exception as e:  # noqa: BLE001
                # exc_info : sans la pile, une erreur de pipeline est
                # indéboguable — le message seul ne dit pas d'où elle vient.
                log.error("échec après %.1f s : %s", time.perf_counter() - debut, e,
                          exc_info=True)
                loop.call_soon_threadsafe(q.put_nowait, ("error", {"message": str(e)}))
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)  # sentinelle de fin

    loop.run_in_executor(None, run)

    yield _sse("status", {"step": "start", "question": question, "provider": provider, "tts": tts})

    while True:
        item = await q.get()
        if item is None:
            break
        event, data = item
        if event == "result":
            trace      = data.get("trace", {})

            # Mémorisation du tour, en français (langue pivot du pipeline).
            memoire = None
            if conv is not None:
                vide = conv.add_exchange(data.get("question_fr") or question,
                                         data.get("response_fr") or "")
                memoire = {
                    "count": len(conv.messages),
                    "max":   conv.snapshot()["max"],
                    "reset": vide,
                }

            audio_path = data.get("audio")
            audio_url  = (
                f"/static/{os.path.basename(audio_path)}"
                if audio_path and os.path.exists(audio_path) else None
            )
            yield _sse("result", {
                "response":    data.get("response", ""),
                "response_fr": data.get("response_fr", ""),
                "response_wo": data.get("response_wo"),
                "qr_code":     data.get("qr_code"),
                "audio_url":   audio_url,
                "lang":        trace.get("input_lang"),
                # Traçabilité RAG, au premier niveau : le client n'a pas à
                # fouiller `trace`, qui est un objet de débogage volumineux et
                # instable — et qui ne doit pas être relayé au navigateur.
                "sources":     data.get("sources", []),
                "memory":      memoire,
                "trace":       trace,
            })
        else:
            # Le pipeline ne connaît que des chemins de fichiers ; le client
            # n'a rien à faire d'un chemin absolu sur le disque du serveur.
            if data.get("step") == "tts_chunk":
                chemin = data.pop("audio", None)
                data["audio_url"] = (
                    f"/static/{os.path.basename(chemin)}"
                    if chemin and os.path.exists(chemin) else None
                )
            yield _sse(event, data)

    yield _sse("done", {})


async def _audio_pipeline_sse(tmp_path: str, provider: str, tts: bool, tts_out: str,
                              lang: str = "wo", session_id: str | None = None,
                              organization_id: str | None = None):
    """Variante audio : STT dans la langue demandée, puis pipeline SSE.

    `lang` vient du bouton pressé sur la borne ('wo' ou 'fr') : il choisit le
    moteur STT et sert ensuite d'indice de langue au pipeline, ce qui évite une
    détection automatique inutile — et parfois fausse — sur la transcription.
    """
    loop = asyncio.get_running_loop()
    yield _sse("status", {"step": "stt", "lang": lang})
    try:
        from input.stt import transcribe
        text = await loop.run_in_executor(None, lambda: transcribe(tmp_path, language=lang))
    except Exception as e:  # noqa: BLE001
        yield _sse("error", {"message": f"STT échoué : {e}"})
        yield _sse("done", {})
        return

    if not text.strip():
        yield _sse("error", {"message": "Aucun texte transcrit"})
        yield _sse("done", {})
        return

    async for chunk in _pipeline_sse(text, provider, tts, tts_out,
                                     lang_hint=lang, session_id=session_id,
                                     organization_id=organization_id):
        yield chunk


# =============================================================================
#  Organisation — validation commune
# =============================================================================

def _organisation(valeur: str | None) -> str:
    """Valide l'`organization_id` et retourne sa forme canonique, ou 400.

    Le service n'a aucune authentification : cet identifiant arrive tel quel
    depuis le réseau et sert ensuite de segment de chemin sur le disque. Il est
    donc validé ici, en entrée, avant d'atteindre quoi que ce soit.
    """
    try:
        return vectorstore.valider_organisation(valeur)
    except vectorstore.OrganisationInvalide as e:
        raise HTTPException(400, str(e)) from None


# =============================================================================
#  Endpoints
# =============================================================================

@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/borne")
def borne():
    """Interface borne : simulation 3D (three.js) d'un terminal physique à deux
    boutons — commande vocale on/off et start/stop."""
    return FileResponse(os.path.join(STATIC_DIR, "borne.html"))


@app.get("/borne/simple")
def borne_simple():
    """Même borne, châssis dessiné en CSS : repli sans WebGL ni CDN."""
    return FileResponse(os.path.join(STATIC_DIR, "borne-simple.html"))


@app.get("/admin")
def admin():
    """Console d'administration documentaire.

    La page est servie telle quelle : elle demande l'`organization_id` à
    l'usager et retient le dernier utilisé. Aucun identifiant n'est injecté ici
    — le service ne connaît pas le registre des organisations, c'est le backend
    qui le détient.
    """
    return FileResponse(os.path.join(STATIC_DIR, "admin.html"))


@app.get("/health")
def health():
    return {
        "status":       "ok",
        "version":      "3.1.0",
        "llm_provider": settings.LLM_PROVIDER,
        "llm_ready":    settings.llm_ready,
        "tts":          "oolel-voices",
        "nllb_model":   settings.NLLB_WO_FR_MODEL,
        "stt_wo":       settings.STT_WO_MODEL,
        "stt_fr":       settings.STT_FR_MODEL,
        "reranker":     settings.RERANKER_MODEL,
        "device":       device.infos(),
        # Les comptages sont désormais propres à chaque organisation : un total
        # global n'aurait aucun sens, et laisserait croire à une base partagée.
        "organizations": {
            "total":    len(vectorstore.organisations()),
            "chargees": vectorstore.organisations_chargees(),
        },
        "storage":      storage.sante(),
        "memory": {
            "enabled":      settings.MEMORY_ENABLED,
            "max_messages": settings.MEMORY_MAX_MESSAGES,
            "sessions":     memory.sessions(),
        },
    }


# ── Texte ─────────────────────────────────────────────────────────────────

@app.post("/ask")
async def ask(req: AskRequest):
    """Question texte (FR ou WO) — réponse en SSE (text/event-stream).

    REST pour l'envoi (ce POST), SSE pour la réponse. Le pipeline détecte la
    langue, traduit si WO, exécute le RAG complet (hybrid → MMR → reranker),
    génère la réponse, re-traduit si WO, et produit optionnellement un audio TTS
    (renvoyé via `audio_url`, servi sur /static).
    """
    if not req.question.strip():
        raise HTTPException(400, "Question vide")

    organization_id = _organisation(req.organization_id)
    provider = req.provider or settings.LLM_PROVIDER
    tts_out  = os.path.join(STATIC_DIR, f"response_{uuid.uuid4().hex[:8]}.wav")
    return StreamingResponse(
        _pipeline_sse(req.question.strip(), provider, req.tts, tts_out,
                      lang_hint=req.lang, session_id=req.session_id,
                      organization_id=organization_id),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


# ── Audio ─────────────────────────────────────────────────────────────────

@app.post("/ask/audio")
async def ask_audio(
    file:            UploadFile = File(...),
    organization_id: str | None = Form(None),
    tts:             bool       = Form(False),
    provider:        str | None = Form(None),
    lang:            str        = Form("wo"),
    session_id:      str | None = Form(None),
):
    """Audio (WAV, MP3, M4A, WebM) → STT → pipeline RAG → réponse SSE + TTS optionnel.

    `lang` sélectionne le moteur STT : 'wo' → soynade-research/Wolof-HuBERT-CTC
    (décodage CTC), 'fr' → Whisper (STT_FR_MODEL). Cette même langue est passée
    au pipeline comme indice, la réponse est streamée en SSE.
    """
    organization_id = _organisation(organization_id)

    # Sauvegarde du fichier uploadé
    safe_name = os.path.basename(file.filename or "audio.wav")
    tmp_path  = os.path.join(UPLOAD_DIR, safe_name)
    with open(tmp_path, "wb") as f:
        f.write(await file.read())

    provider = provider or settings.LLM_PROVIDER
    tts_out  = os.path.join(STATIC_DIR, f"response_{uuid.uuid4().hex[:8]}.wav")
    return StreamingResponse(
        _audio_pipeline_sse(tmp_path, provider, tts, tts_out, lang=lang,
                            session_id=session_id,
                            organization_id=organization_id),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


# ── Mémoire conversationnelle ─────────────────────────────────────────────

@app.get("/session/{session_id}")
def get_session(session_id: str):
    """Tableau de messages de la session — [{role, content}]."""
    snap = memory.snapshot(session_id)
    if snap is None:
        return {"session_id": session_id, "messages": [], "count": 0,
                "max": settings.MEMORY_MAX_MESSAGES, "resets": 0, "total": 0}
    return snap


@app.post("/session/{session_id}/reset")
def reset_session(session_id: str):
    """Vide la mémoire sans supprimer la session."""
    return {"ok": True, "existed": memory.reset(session_id)}


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    """Oublie la session (fin de session sur la borne)."""
    return {"ok": True, "existed": memory.forget(session_id)}


# ── Traduction ────────────────────────────────────────────────────────────

@app.get("/translate")
def translate_test(
    text:      str = "Jërejëf lool ci dimbal bi.",
    direction: str = "wo2fr",
):
    """Test rapide de traduction NLLB (sans LLM)."""
    from translation.nllb import wolof_to_french, french_to_wolof
    if direction == "fr2wo":
        result, seconds = french_to_wolof(text)
        dir_label = "fr→wo"
    else:
        result, seconds = wolof_to_french(text)
        dir_label = "wo→fr"
    return {
        "input":     text,
        "output":    result,
        "direction": dir_label,
        "model":     settings.NLLB_WO_FR_MODEL,
        "seconds":   seconds,
    }


# ── Audio TTS ─────────────────────────────────────────────────────────────
#  L'audio généré est écrit dans STATIC_DIR sous un nom unique et servi via le
#  montage /static. La réponse SSE renvoie son URL dans le champ `audio_url`.


# ── Admin RAG — scopé par organisation ────────────────────────────────────
#
#  L'`organization_id` est dans le CHEMIN, pas dans le corps. Ce n'est pas
#  cosmétique : un segment de chemin oublié donne un 404, alors qu'un champ de
#  corps oublié donnerait une opération silencieusement globale. Sur une API
#  sans authentification, la différence entre un 404 et « effacer les documents
#  de tous les clients » vaut la verbosité de l'URL.

def _resultat_ingestion(org: str, r: dict) -> dict:
    return {
        "ok":              True,
        "document_id":     r["document_id"],
        "organization_id": org,
        "title":           r["title"],
        "category":        r.get("category"),
        "chunks":          r["chunks"],
        "replaced":        r["replaced"],
        "added":           r["added"],
    }


@app.post("/admin/organizations/{organization_id}/documents/from-storage")
async def add_document_from_storage(organization_id: str, ptr: DocumentPointeur):
    """Ingère un document déjà déposé sur le stockage objet (MinIO).

    Voie recommandée : le backend dépose le fichier puis notifie l'IA avec un
    pointeur (bucket + objectKey). L'IA télécharge l'objet, l'indexe, et rend
    l'identifiant du document.

    Rejouer cet appel est sans danger : le remplacement par `documentId` rend
    l'opération idempotente — un appel répété produit le même état final, pas
    des doublons.
    """
    org = _organisation(organization_id)
    if not ptr.documentId.strip():
        raise HTTPException(400, "documentId requis")

    loop = asyncio.get_running_loop()
    try:
        chemin = await loop.run_in_executor(
            None, lambda: storage.telecharger(ptr.bucket, ptr.objectKey, UPLOAD_DIR)
        )
    except storage.ObjetIntrouvable as e:
        raise HTTPException(404, str(e)) from None
    except (storage.ObjetTropVolumineux, ValueError) as e:
        raise HTTPException(400, str(e)) from None
    except storage.StockageNonConfigure as e:
        # 500 et pas 502 : rejouer n'y changera rien, c'est un défaut de
        # déploiement côté IA. Le backend doit alerter, pas réessayer en boucle.
        raise HTTPException(500, str(e)) from None
    except storage.StockageIndisponible as e:
        # 502 : la panne est en aval, chez le stockage. C'est ce qui dit au
        # backend de rejouer avec backoff plutôt que d'alerter.
        raise HTTPException(502, str(e)) from None

    try:
        r = await loop.run_in_executor(None, lambda: ingestion.ingest_file(
            org, chemin,
            title       = ptr.title or os.path.basename(ptr.objectKey),
            document_id = ptr.documentId.strip(),
            category    = ptr.category,
            source      = ptr.objectKey,      # la provenance réelle, pas le fichier temporaire
        ))
    except ingestion.ExtractionImpossible as e:
        raise HTTPException(400, str(e)) from None
    except vectorstore.IndexIncompatible as e:
        raise HTTPException(409, str(e)) from None
    except RuntimeError as e:
        raise HTTPException(500, str(e)) from None
    finally:
        # Le fichier téléchargé ne sert qu'à l'extraction : le garder ferait
        # grossir uploads/ d'une copie de chaque document ingéré.
        try:
            os.unlink(chemin)
        except OSError:
            pass

    if r["chunks"] == 0:
        # Cas le plus fréquent en production : un PDF scanné, donc une image.
        # Le message doit permettre à l'agent qui a déposé le fichier de
        # comprendre ça, et pas de conclure que « ça a planté ».
        raise HTTPException(
            400,
            "Aucun texte exploitable dans ce fichier. S'il s'agit d'un PDF "
            "scanné (une image), il doit d'abord passer par une reconnaissance "
            "de caractères."
        )
    return _resultat_ingestion(org, r)


@app.post("/admin/organizations/{organization_id}/documents")
async def add_document(
    organization_id: str,
    file:        UploadFile | None = File(None),
    text:        str | None        = Form(None),
    title:       str | None        = Form(None),
    document_id: str | None        = Form(None),
    category:    str | None        = Form(None),
):
    """Ingestion directe d'un document (TXT, MD, PDF) ou de texte brut.

    Conservée pour les tests et les petits contenus ; la voie normale en
    production est `from-storage`.

    Le document est découpé en fragments sémantiques (512 caractères, 80 de
    recouvrement), enrichi de métadonnées, encodé et indexé dans la base de
    CETTE organisation.
    """
    org = _organisation(organization_id)
    if file is None and (text is None or not text.strip()):
        raise HTTPException(400, "Fournissez un fichier ou du texte")

    if file is not None:
        ext = os.path.splitext(file.filename or "")[1].lower()
        if ext not in {".txt", ".md", ".pdf"}:
            raise HTTPException(400, f"Format non supporté : {ext} (accepté : txt, md, pdf)")

        # Sécurisation du nom de fichier (évite path traversal)
        safe_name = os.path.basename(file.filename)
        path      = os.path.join(UPLOAD_DIR, safe_name)
        with open(path, "wb") as f:
            f.write(await file.read())

        try:
            r = ingestion.ingest_file(org, path, title=title,
                                      document_id=document_id, category=category)
        except ingestion.ExtractionImpossible as e:
            raise HTTPException(400, str(e))
        except vectorstore.IndexIncompatible as e:
            raise HTTPException(409, str(e))
        except RuntimeError as e:
            raise HTTPException(500, str(e))

        if r["chunks"] == 0:
            raise HTTPException(400, "Aucun texte exploitable dans ce fichier")
        return _resultat_ingestion(org, r)

    # Texte brut
    try:
        r = ingestion.ingest_text(org, text, title or "Document manuel",
                                  document_id=document_id, category=category)
    except vectorstore.IndexIncompatible as e:
        raise HTTPException(409, str(e))
    if r["chunks"] == 0:
        raise HTTPException(400, "Texte vide")
    return _resultat_ingestion(org, r)


@app.get("/admin/organizations/{organization_id}/documents")
def list_documents(organization_id: str):
    """Documents de cette organisation.

    Une organisation inconnue renvoie une liste vide, PAS un 404 : elle n'existe
    simplement pas encore, et sa base sera créée au premier document ingéré.
    """
    org = _organisation(organization_id)
    try:
        documents = vectorstore.all_documents(org)
        total     = vectorstore.count(org)
    except vectorstore.IndexIncompatible as e:
        raise HTTPException(409, str(e)) from None
    return {
        "organization_id": org,
        "documents":       documents,
        "total_documents": len(documents),
        "total_chunks":    total,
    }


@app.delete("/admin/organizations/{organization_id}/documents/{document_id}")
def remove_document(organization_id: str, document_id: str):
    org = _organisation(organization_id)
    n   = vectorstore.delete_document(org, document_id)
    if n == 0:
        raise HTTPException(404, "Document introuvable dans cette organisation")
    return {"ok": True, "organization_id": org, "deleted_chunks": n}


@app.post("/admin/organizations/{organization_id}/documents/clear")
def clear_documents(organization_id: str):
    """Efface tous les documents de CETTE organisation. Les autres sont intactes."""
    org = _organisation(organization_id)
    n   = vectorstore.clear_all(org)
    return {"ok": True, "organization_id": org, "deleted_chunks": n}


@app.get("/admin/organizations")
def list_organizations():
    """Organisations présentes sur le disque — exploitation et supervision."""
    orgs = vectorstore.organisations()
    return {"organizations": orgs, "total": len(orgs),
            "chargees": vectorstore.organisations_chargees()}


@app.delete("/admin/organizations/{organization_id}")
def remove_organization(organization_id: str):
    """Supprime la base d'une organisation, répertoire compris. IRRÉVERSIBLE.

    À câbler sur la résiliation d'un client : c'est la garantie d'effacement qui
    se démontre — il ne reste rien sur le disque, pas des lignes filtrées.
    """
    org = _organisation(organization_id)
    r   = vectorstore.delete_organization(org)
    if not r["existed"]:
        raise HTTPException(404, "Organisation inconnue")
    return {"ok": True, "organization_id": org,
            "deleted_documents": r["deleted_documents"],
            "deleted_chunks":    r["deleted_chunks"]}


# ── Évaluation RAGAS ─────────────────────────────────────────────────────

@app.post("/eval/ragas")
def eval_ragas(req: RagasRequest):
    """Lance l'évaluation RAGAS sur un jeu de test fourni en JSON.

    Body :
    {
      "provider": "groq",
      "test_cases": [
        {
          "question": "Quels documents pour une CNI ?",
          "reference_answer": "Extrait de naissance, photos...",
          "category": "procedure",
          "language": "fr"
        }
      ]
    }

    Retourne les métriques agrégées + détail par cas.
    """
    from evaluation.ragas_eval import evaluate_pipeline, TestCase
    from dataclasses import asdict

    cases = []
    for tc in req.test_cases:
        if "question" not in tc or "reference_answer" not in tc:
            raise HTTPException(400, "Chaque cas doit avoir 'question' et 'reference_answer'")
        cases.append(TestCase(
            question         = tc["question"],
            reference_answer = tc["reference_answer"],
            category         = tc.get("category", "procedure"),
            language         = tc.get("language", "fr"),
        ))

    output_path = os.path.join(BASE_DIR, "resultats_ragas", "evaluation_v3.json")
    summary     = evaluate_pipeline(cases, provider=req.provider, output_path=output_path)

    return {
        "summary":    asdict(summary),
        "output_file": output_path,
    }


# =============================================================================
#  Démarrage
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    # HTTPS si un certificat est configuré : indispensable pour que le micro
    # soit accessible ailleurs que sur localhost (cf. SSL_CERTFILE dans .env).
    ssl_options = {}
    if settings.ssl_enabled:
        ssl_options = {
            "ssl_certfile": settings.SSL_CERTFILE,
            "ssl_keyfile":  settings.SSL_KEYFILE,
        }
        journal("serveur").info("HTTPS activé → https://%s:%s",
                                settings.HOST, settings.PORT)

    # log_config=None : uvicorn installe sinon ses propres gestionnaires, et ses
    # lignes (démarrage, accès HTTP) sortiraient dans un autre format, sans
    # horodatage à la milliseconde ni identifiant de requête. En le désactivant,
    # tout passe par le journal du projet — une seule trace à lire.
    configurer_journal()
    uvicorn.run(app, host=settings.HOST, port=settings.PORT,
                log_config=None, **ssl_options)
