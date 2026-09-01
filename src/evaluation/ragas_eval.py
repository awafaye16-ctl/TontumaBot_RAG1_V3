"""Évaluation RAGAS du système RAG — V3.

Métriques calculées :

  Retrieval (qualité des chunks récupérés) :
    context_precision  — les chunks pertinents apparaissent-ils en tête ?
    context_recall     — tous les faits nécessaires sont-ils dans les chunks ?

  LLM (qualité de la réponse générée) :
    faithfulness       — la réponse est-elle supportée par le contexte ?
    answer_relevance   — la réponse répond-elle réellement à la question ?
    answer_correctness — correspond-elle à une réponse de référence ?

  Système :
    latency_retrieval_ms
    latency_rerank_ms
    latency_llm_ms
    latency_total_ms

Usage minimal :
    from evaluation.ragas_eval import evaluate_pipeline
    results = evaluate_pipeline(test_cases)
    print(results.summary())
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# =============================================================================
#  Structures de données
# =============================================================================

@dataclass
class TestCase:
    """Un cas de test pour l'évaluation RAGAS."""
    question: str                          # question de l'utilisateur
    reference_answer: str                  # réponse attendue (vérité terrain)
    language: str = "fr"                   # "fr" ou "wo"
    category: str = "procedure"            # "procedure" | "orientation" | "absente"


@dataclass
class EvalResult:
    """Résultat d'évaluation pour un cas de test."""
    question: str
    category: str
    language: str

    # Réponses
    generated_answer: str = ""
    reference_answer: str  = ""

    # Contexte récupéré
    retrieved_chunks: list[str] = field(default_factory=list)
    n_chunks: int = 0

    # ── Métriques retrieval ───────────────────────────────────────────────
    context_precision: Optional[float] = None  # chunks pertinents bien classés
    context_recall:    Optional[float] = None  # couverture des faits nécessaires

    # ── Métriques LLM ────────────────────────────────────────────────────
    faithfulness:       Optional[float] = None  # réponse supportée par le contexte
    answer_relevance:   Optional[float] = None  # réponse pertinente par rapport à la question
    answer_correctness: Optional[float] = None  # concordance avec la référence

    # ── Métriques système ────────────────────────────────────────────────
    latency_retrieval_ms: Optional[float] = None
    latency_rerank_ms:    Optional[float] = None
    latency_llm_ms:       Optional[float] = None
    latency_total_ms:     Optional[float] = None

    # Abstention correcte (cas "absente")
    abstention_correct: Optional[bool] = None


@dataclass
class EvalSummary:
    """Résumé agrégé d'une session d'évaluation."""
    n_cases: int = 0
    mean_context_precision:  Optional[float] = None
    mean_context_recall:     Optional[float] = None
    mean_faithfulness:       Optional[float] = None
    mean_answer_relevance:   Optional[float] = None
    mean_answer_correctness: Optional[float] = None
    mean_latency_total_ms:   Optional[float] = None
    abstention_rate:         Optional[float] = None  # taux abstention correcte


# =============================================================================
#  Métriques locales (sans appel LLM externe)
# =============================================================================

def _normalize(text: str) -> str:
    import re, unicodedata
    text = str(text).lower().strip()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _word_f1(reference: str, hypothesis: str) -> float:
    """F1 au niveau des mots (insensible à l'ordre)."""
    r = set(_normalize(reference).split())
    h = set(_normalize(hypothesis).split())
    if not r or not h:
        return 0.0
    common    = r & h
    precision = len(common) / len(h)
    recall    = len(common) / len(r)
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def _is_abstention(response: str) -> bool:
    """Détecte si le modèle a correctement refusé de répondre."""
    r = _normalize(response)
    keywords = [
        "information_absente", "information absente",
        "pas mentionne", "ne mentionne pas",
        "aucune information", "je ne sais pas",
        "pas dans le contexte", "ne figure pas",
    ]
    return any(k in r.replace(" ", "_") or k in r for k in keywords)


def _context_precision(question: str, chunks: list[str], reference: str) -> float:
    """Proportion de chunks pertinents parmi les N premiers.

    Un chunk est considéré pertinent si son score F1 avec la référence > 0.1.
    context_precision@K = (chunks_pertinents_dans_top_K) / K
    """
    if not chunks:
        return 0.0
    scores    = [_word_f1(reference, c) for c in chunks]
    relevant  = [s > 0.1 for s in scores]
    precision = sum(relevant) / len(chunks)
    return round(precision, 4)


def _context_recall(question: str, chunks: list[str], reference: str) -> float:
    """Proportion des mots de la référence couverts par les chunks.

    context_recall = mots_référence_présents_dans_contexte / mots_référence_total
    """
    if not chunks or not reference.strip():
        return 0.0
    ref_words   = set(_normalize(reference).split())
    ctx_words   = set(_normalize(" ".join(chunks)).split())
    if not ref_words:
        return 0.0
    covered = len(ref_words & ctx_words) / len(ref_words)
    return round(covered, 4)


def _faithfulness(response: str, chunks: list[str]) -> float:
    """Mesure si chaque phrase de la réponse est supportée par le contexte.

    faithfulness = phrases_supportées / phrases_totales
    Une phrase est supportée si son score F1 avec un chunk > seuil.
    """
    import re
    if not chunks or not response.strip():
        return 0.0
    sentences = [s.strip() for s in re.split(r"[.!?]\s+", response) if len(s.strip()) > 10]
    if not sentences:
        return 0.0
    context      = " ".join(chunks)
    supported    = sum(1 for s in sentences if _word_f1(s, context) > 0.15)
    return round(supported / len(sentences), 4)


def _answer_relevance(question: str, response: str) -> float:
    """Mesure si la réponse répond à la question (par recouvrement de mots-clés)."""
    return _word_f1(question, response)


def _answer_correctness(response: str, reference: str) -> float:
    """Concordance entre la réponse générée et la référence humaine (F1 mots)."""
    return _word_f1(reference, response)


# =============================================================================
#  Évaluation d'un seul cas
# =============================================================================

def evaluate_single(
    case: TestCase,
    provider: str = "groq",
    tts: bool     = False,
) -> EvalResult:
    """Évalue un cas de test en appelant le pipeline complet."""
    from pipeline import answer as pipeline_answer

    result = EvalResult(
        question=case.question,
        category=case.category,
        language=case.language,
        reference_answer=case.reference_answer,
    )

    # Appel pipeline
    raw = pipeline_answer(case.question, provider=provider, tts=tts)

    result.generated_answer = raw.get("response", "")
    trace                   = raw.get("trace", {})

    # Chunks récupérés depuis la trace
    retrieval = trace.get("retrieval", {})
    result.retrieved_chunks = [c["text"] for c in retrieval.get("chunks", [])]
    result.n_chunks         = retrieval.get("n_reranked", 0)

    # Latences
    result.latency_retrieval_ms = retrieval.get("latency_hybrid_ms")
    result.latency_rerank_ms    = retrieval.get("latency_rerank_ms")
    result.latency_llm_ms       = trace.get("llm", {}).get("latency_ms")
    result.latency_total_ms     = trace.get("total_latency_ms")

    # Métriques
    if case.category == "absente":
        result.abstention_correct = _is_abstention(result.generated_answer)
    else:
        result.context_precision  = _context_precision(
            case.question, result.retrieved_chunks, case.reference_answer
        )
        result.context_recall     = _context_recall(
            case.question, result.retrieved_chunks, case.reference_answer
        )
        result.faithfulness       = _faithfulness(
            result.generated_answer, result.retrieved_chunks
        )
        result.answer_relevance   = _answer_relevance(
            case.question, result.generated_answer
        )
        result.answer_correctness = _answer_correctness(
            result.generated_answer, case.reference_answer
        )

    return result


# =============================================================================
#  Évaluation batch
# =============================================================================

def evaluate_pipeline(
    test_cases: list[TestCase],
    provider: str   = "groq",
    output_path: str | None = None,
) -> EvalSummary:
    """Évalue tous les cas de test et retourne un résumé agrégé.

    Args:
        test_cases:   Liste de TestCase.
        provider:     LLM à utiliser.
        output_path:  Chemin JSON pour sauvegarder les résultats détaillés.

    Returns:
        EvalSummary avec les moyennes de toutes les métriques.
    """
    results: list[EvalResult] = []
    for i, case in enumerate(test_cases):
        print(f"[RAGAS] {i+1}/{len(test_cases)} — {case.question[:60]}...")
        r = evaluate_single(case, provider=provider)
        results.append(r)

    # Sauvegarde JSON détaillée
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)
        print(f"[RAGAS] Résultats sauvegardés → {output_path}")

    # Agrégation
    def _mean(values: list) -> Optional[float]:
        vals = [v for v in values if v is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    non_absente = [r for r in results if r.category != "absente"]
    absente     = [r for r in results if r.category == "absente"]

    summary = EvalSummary(
        n_cases                  = len(results),
        mean_context_precision   = _mean([r.context_precision  for r in non_absente]),
        mean_context_recall      = _mean([r.context_recall      for r in non_absente]),
        mean_faithfulness        = _mean([r.faithfulness        for r in non_absente]),
        mean_answer_relevance    = _mean([r.answer_relevance    for r in non_absente]),
        mean_answer_correctness  = _mean([r.answer_correctness  for r in non_absente]),
        mean_latency_total_ms    = _mean([r.latency_total_ms    for r in results]),
        abstention_rate          = _mean([float(r.abstention_correct) for r in absente
                                          if r.abstention_correct is not None]),
    )

    # Affichage console
    print("\n" + "=" * 55)
    print("  RAGAS — Résumé d'évaluation V3")
    print("=" * 55)
    print(f"  Cas évalués              : {summary.n_cases}")
    print(f"  Context Precision        : {summary.mean_context_precision}")
    print(f"  Context Recall           : {summary.mean_context_recall}")
    print(f"  Faithfulness             : {summary.mean_faithfulness}")
    print(f"  Answer Relevance         : {summary.mean_answer_relevance}")
    print(f"  Answer Correctness       : {summary.mean_answer_correctness}")
    print(f"  Abstention correcte      : {summary.abstention_rate}")
    print(f"  Latence moyenne totale   : {summary.mean_latency_total_ms} ms")
    print("=" * 55)

    return summary


# =============================================================================
#  Jeu de test de démonstration
# =============================================================================

DEMO_TEST_CASES = [
    TestCase(
        question="Quels documents faut-il pour obtenir une carte d'identité ?",
        reference_answer="Un extrait de naissance, deux photos d'identité et le reçu de paiement.",
        category="procedure",
    ),
    TestCase(
        question="Où déposer une demande d'extrait de naissance ?",
        reference_answer="À la mairie du lieu de naissance, service de l'état civil.",
        category="orientation",
    ),
    TestCase(
        question="Combien coûte un passeport ?",
        reference_answer="25000 francs CFA.",
        category="procedure",
    ),
    TestCase(
        question="Quel est le délai de livraison d'un permis de conduire ?",
        reference_answer="INFORMATION_ABSENTE",
        category="absente",
    ),
    TestCase(
        question="dama beug wout kayitu juddu?",
        reference_answer="Naka laay def ngir am kayitu juddu?",
        language="wo",
        category="procedure",
    ),
]


if __name__ == "__main__":
    summary = evaluate_pipeline(
        DEMO_TEST_CASES,
        provider="groq",
        output_path="resultats_ragas/evaluation_v3.json",
    )
