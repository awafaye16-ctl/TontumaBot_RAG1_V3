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

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import settings
from pipeline import answer as pipeline_answer
from seed_docs import get_documents
import vectorstore
import ingestion

app = FastAPI(title="TontumaBot V3", version="3.0.0")

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
    question:   str
    provider:   str | None = None   # groq | gemini | local
    tts:        bool        = False
    tts_engine: str | None = None   # oolel | speecht5 | edge


class RagasRequest(BaseModel):
    """Jeu de test pour l'évaluation RAGAS."""
    test_cases: list[dict]   # liste de {question, reference_answer, category?, language?}
    provider:   str = "groq"


# =============================================================================
#  Endpoints
# =============================================================================

@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/admin")
def admin():
    return FileResponse(os.path.join(STATIC_DIR, "admin.html"))


@app.get("/health")
def health():
    return {
        "status":       "ok",
        "version":      "3.0.0",
        "llm_provider": settings.LLM_PROVIDER,
        "llm_ready":    settings.llm_ready,
        "tts_engine":   settings.TTS_ENGINE,
        "nllb_wo_fr":   settings.NLLB_WO_FR_MODEL,
        "nllb_fr_wo":   settings.NLLB_FR_WO_MODEL,
        "stt_model":    settings.STT_MODEL_PATH,
        "reranker":     settings.RERANKER_MODEL,
        "n_documents":  len(vectorstore.all_documents()),
        "n_chunks":     vectorstore.count(),
    }


# ── Texte ─────────────────────────────────────────────────────────────────

@app.post("/ask")
def ask(req: AskRequest):
    """Question texte (FR ou WO).

    Le pipeline détecte la langue, traduit si WO, exécute le RAG complet
    (hybrid → MMR → reranker), génère la réponse, re-traduit si WO,
    et produit optionnellement un audio TTS.
    """
    if not req.question.strip():
        raise HTTPException(400, "Question vide")

    provider = req.provider or settings.LLM_PROVIDER
    return pipeline_answer(
        req.question.strip(),
        provider    = provider,
        tts         = req.tts,
        tts_engine  = req.tts_engine,
        tts_out     = os.path.join(STATIC_DIR, "response.wav"),
        seed_docs   = SEED_DOCS,
        seed_filtered = SEED_FILTERED,
    )


# ── Audio ─────────────────────────────────────────────────────────────────

@app.post("/ask/audio")
async def ask_audio(
    file:       UploadFile     = File(...),
    tts:        bool           = Form(False),
    tts_engine: str | None     = Form(None),
    provider:   str | None     = Form(None),
):
    """Audio (WAV, MP3, M4A, WebM) → STT → pipeline RAG → réponse + TTS optionnel.

    Le STT utilise M9and2M/whisper-small-wolof.
    La langue est détectée automatiquement sur la transcription.
    """
    # Sauvegarde du fichier uploadé
    safe_name = os.path.basename(file.filename or "audio.wav")
    tmp_path  = os.path.join(UPLOAD_DIR, safe_name)
    with open(tmp_path, "wb") as f:
        f.write(await file.read())

    # Transcription STT
    try:
        from input.stt import transcribe
        text = transcribe(tmp_path, language=settings.STT_LANGUAGE or None)
    except Exception as e:
        raise HTTPException(500, f"STT échoué : {e}")

    if not text.strip():
        raise HTTPException(400, "Aucun texte transcrit")

    return pipeline_answer(
        text,
        provider      = provider or settings.LLM_PROVIDER,
        tts           = tts,
        tts_engine    = tts_engine,
        tts_out       = os.path.join(STATIC_DIR, "response.wav"),
        seed_docs     = SEED_DOCS,
        seed_filtered = SEED_FILTERED,
    )


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
        model = settings.NLLB_FR_WO_MODEL
        dir_label = "fr→wo"
    else:
        result, seconds = wolof_to_french(text)
        model = settings.NLLB_WO_FR_MODEL
        dir_label = "wo→fr"
    return {
        "input":     text,
        "output":    result,
        "direction": dir_label,
        "model":     model,
        "seconds":   seconds,
    }


# ── Audio TTS ─────────────────────────────────────────────────────────────

@app.get("/response.wav")
def get_audio():
    path = os.path.join(STATIC_DIR, "response.wav")
    if not os.path.exists(path):
        raise HTTPException(404, "Aucun audio généré")
    return FileResponse(path, media_type="audio/wav")


# ── Admin RAG ─────────────────────────────────────────────────────────────

@app.post("/admin/documents")
async def add_document(
    file:  UploadFile | None = File(None),
    text:  str | None        = Form(None),
    title: str | None        = Form(None),
):
    """Ingère un document (TXT, MD, PDF) ou du texte brut dans ChromaDB.

    Le document est automatiquement :
      - découpé en chunks sémantiques (512 chars, overlap 80)
      - enrichi de métadonnées (document_id, titre, source, index, nb_mots, date)
      - encodé en embeddings et indexé dans ChromaDB
    """
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
            n = ingestion.ingest_file(path, title=title)
        except RuntimeError as e:
            raise HTTPException(500, str(e))

        if n == 0:
            raise HTTPException(400, "Aucun texte exploitable dans ce fichier")
        return {"ok": True, "chunks": n, "title": title or safe_name}

    # Texte brut
    n = ingestion.ingest_text(text, title or "Document manuel")
    if n == 0:
        raise HTTPException(400, "Texte vide")
    return {"ok": True, "chunks": n, "title": title or "Document manuel"}


@app.get("/admin/documents")
def list_documents():
    return {
        "documents":    vectorstore.all_documents(),
        "total_chunks": vectorstore.count(),
    }


@app.delete("/admin/documents/{document_id}")
def remove_document(document_id: str):
    n = vectorstore.delete_document(document_id)
    if n == 0:
        raise HTTPException(404, "Document introuvable")
    return {"ok": True, "deleted_chunks": n}


@app.post("/admin/documents/clear")
def clear_documents():
    n = vectorstore.clear_all()
    return {"ok": True, "deleted_chunks": n}


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
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
