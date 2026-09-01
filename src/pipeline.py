"""Pipeline V3 — RAG rigoureux avec évaluation RAGAS.

Flux complet :
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Entrée (texte ou audio)                                            │
  │    ↓ STT si audio  [M9and2M/whisper-small-wolof]                    │
  │  Texte brut (FR ou WO)                                              │
  │    ↓ Détection de langue                                            │
  │  Si WO → traduction WO→FR  [bilalfaye/nllb]                        │
  │    ↓                                                                │
  │  Router d'intention  (procedure | orientation)                      │
  │    ↓                                                                │
  │  ┌── RAG Pipeline ───────────────────────────────────────────────┐  │
  │  │  1. Hybrid search  BM25 + vectoriel  → fetch_k=20 candidats   │  │
  │  │  2. MMR            diversité          → 8 chunks retenus       │  │
  │  │  3. Reranker       cross-encoder      → top_k=3 envoyés au LLM │  │
  │  └───────────────────────────────────────────────────────────────┘  │
  │    ↓                                                                │
  │  LLM  [Groq qwen3-32b | Gemini | local Qwen2.5-7B]                 │
  │    ↓                                                                │
  │  Si WO → traduction FR→WO  [Lahad/nllb]                            │
  │    ↓ TTS si demandé  [Oolel-Voices → SpeechT5 → edge-tts]         │
  │  Réponse JSON  { response, response_fr, response_wo?, audio? }     │
  └─────────────────────────────────────────────────────────────────────┘

Trace complète :
  - langue détectée, traductions (modèle + durée)
  - intention, type de recherche
  - chunks candidats, MMR sélectionnés, scores reranker
  - contexte envoyé au LLM
  - TTS utilisé
  - métriques de latence par étape
"""
import re
import time
from typing import Optional

import vectorstore
from language.detector import detect_language
from translation.nllb import wolof_to_french, french_to_wolof
from intent.router import detect_intent
from retrieval.reranker import rerank
from generation.llm import generate


# =============================================================================
#  Nettoyage markdown avant traduction NLLB
# =============================================================================

def _strip_markdown(text: str) -> str:
    """Retire le markdown de la réponse FR avant traduction vers le wolof.

    NLLB ne gère pas les balises markdown (**gras**, # titres, listes à puces…)
    ce qui produit un wolof cassé ou tronqué.
    On conserve la structure textuelle (numéros de liste, ponctuation).
    """
    # Titres markdown
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Gras et italique **..** *..* __..__ _.._ 
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
    # Listes à puces (remplacer par tiret simple)
    text = re.sub(r"^\s*[-*+•]\s+", "- ", text, flags=re.MULTILINE)
    # Liens markdown [texte](url)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    # Code inline `...`
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Lignes vides multiples → une seule
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# =============================================================================
#  RAG — Retrieval + MMR + Reranking
# =============================================================================

def _rag_pipeline(
    question_fr: str,
    intent: str,
    fetch_k: int  = 20,
    mmr_k: int    = 8,
    top_k: int    = 3,
) -> tuple[str, dict]:
    """Exécute les 3 étapes du retrieval et retourne (contexte_fr, retrieval_trace).

    1. Hybrid search (BM25 + vectoriel) → fetch_k candidats
    2. MMR → mmr_k chunks diversifiés
    3. Reranker cross-encoder → top_k envoyés au LLM

    Returns:
        contexte_fr    — chaîne de texte des top_k chunks
        retrieval_trace — dict de traçabilité détaillé
    """
    t0 = time.perf_counter()

    # ── Étape 1 : Hybrid search ───────────────────────────────────────────
    t_hybrid_start = time.perf_counter()
    candidates = vectorstore.hybrid_search(question_fr, k=fetch_k)
    t_hybrid   = round((time.perf_counter() - t_hybrid_start) * 1000, 1)

    if not candidates:
        return "", {
            "n_candidates": 0, "n_mmr": 0, "n_reranked": 0,
            "latency_hybrid_ms": t_hybrid, "latency_mmr_ms": 0,
            "latency_rerank_ms": 0, "chunks": [],
        }

    # ── Étape 2 : MMR ─────────────────────────────────────────────────────
    t_mmr_start  = time.perf_counter()
    mmr_results  = vectorstore.mmr_search(question_fr, k=mmr_k, fetch_k=fetch_k)
    t_mmr        = round((time.perf_counter() - t_mmr_start) * 1000, 1)

    mmr_texts = [r[1] for r in mmr_results]

    # ── Étape 3 : Reranker cross-encoder ──────────────────────────────────
    t_rerank_start = time.perf_counter()
    reranked       = rerank(question_fr, mmr_texts, top_k=top_k)
    t_rerank       = round((time.perf_counter() - t_rerank_start) * 1000, 1)

    # Contexte final pour le LLM
    context_fr = "\n\n".join(doc for _, _, doc in reranked)

    retrieval_trace = {
        "n_candidates":       len(candidates),
        "n_mmr":              len(mmr_results),
        "n_reranked":         len(reranked),
        "latency_hybrid_ms":  t_hybrid,
        "latency_mmr_ms":     t_mmr,
        "latency_rerank_ms":  t_rerank,
        "latency_total_ms":   round((time.perf_counter() - t0) * 1000, 1),
        "reranker_scores":    [round(s, 4) for _, s, _ in reranked],
        "chunks": [
            {
                "rank":  i + 1,
                "score": round(s, 4),
                "text":  doc[:200] + ("…" if len(doc) > 200 else ""),
            }
            for i, (_, s, doc) in enumerate(reranked)
        ],
    }
    return context_fr, retrieval_trace


# =============================================================================
#  Point d'entrée principal
# =============================================================================

def answer(
    unified_text: str,
    provider: str             = "groq",
    tts: bool                 = False,
    tts_engine: Optional[str] = None,
    tts_out: str              = "response.wav",
    seed_docs: Optional[list[str]]  = None,
    seed_filtered: Optional[list[dict]] = None,
) -> dict:
    """Traite une question et retourne la réponse complète avec trace.

    Args:
        unified_text:   Question en wolof ou français.
        provider:       'groq' | 'gemini' | 'local'
        tts:            Générer une réponse vocale.
        tts_engine:     'oolel' | 'speecht5' | 'edge' | None (→ .env)
        tts_out:        Chemin du fichier audio de sortie.
        seed_docs:      Documents en mémoire (si ChromaDB vide).
        seed_filtered:  Docs orientation pour le seed.

    Returns:
        dict {
            trace:        pipeline complet (langues, retrieval, LLM, TTS, latences)
            response_fr:  réponse en français
            response:     réponse dans la langue de l'utilisateur
            response_wo:  réponse en wolof (si entrée wolof)
            audio:        chemin du fichier audio (si tts=True)
        }
    """
    t_total = time.perf_counter()
    trace   = {}

    # ── 1. Détection de langue ────────────────────────────────────────────
    lang             = detect_language(unified_text)
    trace["input_lang"] = lang

    # ── 2. Traduction WO→FR si nécessaire ────────────────────────────────
    question_fr = unified_text
    if lang == "wo":
        t0 = time.perf_counter()
        question_fr, _ = wolof_to_french(unified_text)
        trace["wolof_to_french"] = {
            "model":   "bilalfaye/nllb-200-distilled-600M-wo-fr-en",
            "result":  question_fr,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    # ── 3. Intention ──────────────────────────────────────────────────────
    intent         = detect_intent(question_fr)
    trace["intent"] = intent

    # ── 4. RAG ────────────────────────────────────────────────────────────
    n_docs = vectorstore.count()

    if n_docs > 0:
        context_fr, retrieval_trace = _rag_pipeline(question_fr, intent)
        retrieval_trace["source"] = "chromadb"
    elif seed_docs:
        # Fallback seed en mémoire (démo sans base persistante)
        context_fr, retrieval_trace = _rag_seed(question_fr, intent, seed_docs, seed_filtered)
        retrieval_trace["source"] = "seed"
    else:
        context_fr      = ""
        retrieval_trace = {"source": "aucun document", "n_candidates": 0}

    trace["retrieval"] = retrieval_trace
    trace["context"]   = context_fr
    trace["n_docs_in_db"] = n_docs

    # ── 5. Génération LLM ─────────────────────────────────────────────────
    t0          = time.perf_counter()
    # plain=True si la réponse sera traduite en wolof (NLLB ne gère pas le markdown)
    response_fr = generate(question_fr, context_fr, provider=provider, plain=(lang == "wo"))
    trace["llm"] = {
        "provider":   provider,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
    }

    response = {
        "trace":       trace,
        "response_fr": response_fr,
        "response":    response_fr,
    }

    # ── 6. Traduction FR→WO ───────────────────────────────────────────────
    if lang == "wo":
        t0 = time.perf_counter()
        # Nettoyer le markdown avant traduction (NLLB ne gère pas les balises)
        response_fr_clean = _strip_markdown(response_fr)
        response_wo, _ = french_to_wolof(response_fr_clean)
        response["response_wo"] = response_wo
        response["response"]    = response_wo
        trace["french_to_wolof"] = {
            "model":      "Lahad/nllb200-francais-wolof",
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    # ── 7. TTS (optionnel) ────────────────────────────────────────────────
    if tts:
        from tts_Ooleil.tts import synthesize, source as tts_source
        t0           = time.perf_counter()
        text_for_tts = response.get("response_wo", response_fr)
        synthesize(text_for_tts, tts_out, engine=tts_engine)
        response["audio"] = tts_out
        trace["tts"] = {
            "engine":     tts_source(tts_engine),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    # ── 8. Latence totale ─────────────────────────────────────────────────
    trace["total_latency_ms"] = round((time.perf_counter() - t_total) * 1000, 1)

    return response


# =============================================================================
#  Fallback seed (sans ChromaDB)
# =============================================================================

def _rag_seed(
    question_fr: str,
    intent: str,
    seed_docs: list[str],
    seed_filtered: Optional[list[dict]],
) -> tuple[str, dict]:
    """RAG en mémoire sur les seed_docs quand ChromaDB est vide."""
    from retrieval.hybrid import HybridRetriever
    from retrieval.filtered import FilteredSearch

    t0 = time.perf_counter()

    if intent == "orientation" and seed_filtered:
        fs      = FilteredSearch(seed_filtered)
        matches = fs.search(question_fr, k=5)
        texts   = [d["text"] for d in matches] if matches else seed_docs[:3]
        stype   = "filtered (seed - orientation)"
    else:
        hybrid  = HybridRetriever(seed_docs)
        ranked  = hybrid.search(question_fr, k=8)
        texts   = [seed_docs[i] for i, _ in ranked] if ranked else seed_docs[:3]
        stype   = "hybride BM25 (seed)"

    # Reranker même sur le seed
    reranked   = rerank(question_fr, texts, top_k=3)
    context_fr = "\n\n".join(doc for _, _, doc in reranked)

    return context_fr, {
        "search_type":   stype,
        "n_candidates":  len(texts),
        "n_reranked":    len(reranked),
        "latency_ms":    round((time.perf_counter() - t0) * 1000, 1),
        "reranker_scores": [round(s, 4) for _, s, _ in reranked],
    }
