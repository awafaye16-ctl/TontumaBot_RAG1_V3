"""Pipeline V3 — RAG rigoureux avec évaluation RAGAS.

Flux complet :
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Entrée (texte ou audio)                                            │
  │    ↓ STT si audio  [wo : Wolof-HuBERT-CTC | fr : Whisper]           │
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
  │    ↓ QR Code si procédure                                           │
  │  Réponse JSON  { response, response_fr, response_wo?, audio?, qr_code? }
  └─────────────────────────────────────────────────────────────────────┘

Trace complète :
  - langue détectée, traductions (modèle + durée)
  - intention, type de recherche
  - chunks candidats, MMR sélectionnés, scores reranker
  - contexte envoyé au LLM
  - TTS utilisé
  - QR Code généré (si procédure)
  - métriques de latence par étape
"""
import re
import time
import base64
from io import BytesIO
from typing import Callable, Optional

from config import settings
import vectorstore
from language.detector import detect_language, contredit
from translation.nllb import wolof_to_french, french_to_wolof
from translation.nombres import en_chiffres, realigner, verifier_nombres
from translation.prononciation import pour_synthese
from intent.router import detect_intent
from retrieval.reranker import rerank
from generation.llm import generate

from journal import journal

_log = journal("pipeline")

try:
    import qrcode
    from PIL import Image
    QR_AVAILABLE = True
except Exception:
    QR_AVAILABLE = False


def _generate_qr_code(text: str) -> str | None:
    """Génère un QR code contenant le texte et retourne une string base64 (PNG)."""
    if not QR_AVAILABLE:
        return None
    try:
        qr = qrcode.QRCode(
            version=None,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=10,
            border=4,
        )
        qr.add_data(text)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        buf = BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


# =============================================================================
#  Question de suite → requête de recherche
# =============================================================================
#  Le LLM reçoit l'historique et comprend « et combien ça coûte ? ». Le moteur
#  de recherche, lui, ne voit que des mots : cherché tel quel, ce fragment ne
#  ramène rien. On lui adjoint donc la dernière question de l'usager, mais
#  uniquement quand la question courante est trop courte pour se suffire —
#  au-delà, la concaténation diluerait une question déjà explicite.

_MOTS_MAX_SUITE = 6


def _sujet_courant(history) -> str | None:
    """Dernière question de l'usager qui porte un sujet.

    On remonte jusqu'à une question assez longue pour se suffire à elle-même :
    s'ancrer sur le message précédent ne marche que si celui-ci n'était pas
    lui-même une question de suite. Dans « extrait de naissance ? » → « et le
    coût ? » → « où aller ? », le sujet reste l'extrait de naissance.
    """
    dernier = None
    for m in reversed(history or []):
        if m.get("role") != "user":
            continue
        contenu = m.get("content", "")
        if dernier is None:
            dernier = contenu
        if len(contenu.split()) > _MOTS_MAX_SUITE:
            return contenu
    return dernier


def _contextualize(question_fr: str, history) -> str:
    """Requête de recherche enrichie du sujet courant si la question est courte."""
    if not history or len(question_fr.split()) > _MOTS_MAX_SUITE:
        return question_fr
    sujet = _sujet_courant(history)
    return f"{sujet} {question_fr}" if sujet else question_fr


# =============================================================================
#  Nettoyage markdown avant traduction NLLB


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
    organization_id: str,
    question_fr: str,
    intent: str,
    fetch_k: int  = 20,
    mmr_k: int    = 8,
    top_k: int    = 3,
    query_embedding: Optional[list[float]] = None,
) -> tuple[str, dict, list[dict]]:
    """Exécute les 3 étapes du retrieval DANS LA BASE DE L'ORGANISATION.

    1. Hybrid search (BM25 + vectoriel) → fetch_k candidats
    2. MMR → mmr_k fragments diversifiés
    3. Reranker cross-encoder → top_k envoyés au LLM

    Returns:
        contexte_fr     — chaîne de texte des top_k fragments
        retrieval_trace — dict de traçabilité détaillé (observabilité)
        sources         — documents ayant nourri la réponse, exposés au client
    """
    t0 = time.perf_counter()

    # Embedding de la requête calculé UNE seule fois puis réutilisé partout
    # (déjà calculé par `answer()` pour le router d'intention, le cas échéant)
    q_emb = (
        query_embedding
        if query_embedding is not None
        else vectorstore.embed_query(question_fr)
    )

    # ── Étape 1 : Hybrid search (un seul passage, index BM25 + embeddings cachés)
    t_hybrid_start = time.perf_counter()
    candidates = vectorstore.hybrid_search(organization_id, question_fr, k=fetch_k,
                                           query_embedding=q_emb)
    t_hybrid   = round((time.perf_counter() - t_hybrid_start) * 1000, 1)

    if not candidates:
        return "", {
            "n_candidates": 0, "n_mmr": 0, "n_reranked": 0,
            "latency_hybrid_ms": t_hybrid, "latency_mmr_ms": 0,
            "latency_rerank_ms": 0, "chunks": [],
        }, []

    # ── Étape 2 : MMR sur les candidats déjà récupérés ────────────────────
    t_mmr_start  = time.perf_counter()
    mmr_results  = vectorstore.mmr_from_candidates(organization_id, candidates,
                                                   q_emb, k=mmr_k)
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
        # L'identifiant du fragment voyage avec le texte : sans lui, impossible
        # de dire APRÈS COUP quel passage du corpus a nourri une réponse — ni de
        # confronter la recherche à un jeu d'évaluation. `rerank` renvoie
        # l'index d'origine dans `mmr_texts`, d'où le détour par `mmr_results`.
        "chunks": [
            {
                "rank":  i + 1,
                "id":    mmr_results[idx][0],
                "score": round(s, 4),
                "text":  doc[:200] + ("…" if len(doc) > 200 else ""),
            }
            for i, (idx, s, doc) in enumerate(reranked)
        ],
    }

    # ── Sources exposées au client ────────────────────────────────────────
    # La trace ci-dessus est un objet d'observabilité : volumineuse, instable,
    # et porteuse d'extraits de documents qu'on ne renvoie pas au navigateur.
    # `sources` en est l'extrait stable et consommable, au premier niveau du
    # résultat. Le `document_id` vient des métadonnées du fragment : c'est
    # l'identifiant fourni par l'appelant à l'ingestion, donc sa clé de jointure
    # — à ne pas confondre avec `chunk_id`, qui change à chaque réindexation.
    sources = []
    for i, (idx, score, _doc) in enumerate(reranked):
        chunk_id, _texte, _s, meta = mmr_results[idx]
        meta = meta or {}
        sources.append({
            "document_id": meta.get("document_id"),
            "title":       meta.get("title"),
            "category":    meta.get("category") or None,
            "chunk_id":    chunk_id,
            "rank":        i + 1,
            # Score brut du cross-encoder : non borné, souvent négatif, non
            # comparable d'une question à l'autre. Il ordonne, il ne note pas.
            "score":       round(float(score), 4),
        })

    return context_fr, retrieval_trace, sources


# =============================================================================
#  Point d'entrée principal
# =============================================================================

def answer(
    unified_text: str,
    organization_id: Optional[str] = None,
    provider: str             = "groq",
    tts: bool                 = False,
    tts_out: str              = "response.wav",
    seed_docs: Optional[list[str]]  = None,
    seed_filtered: Optional[list[dict]] = None,
    progress: Optional[Callable[[str, dict], None]] = None,
    lang_hint: Optional[str] = None,
    history: Optional[list[dict]] = None,
) -> dict:
    """Traite une question et retourne la réponse complète avec trace.

    Args:
        unified_text:   Question en wolof ou français.
        organization_id: UUID de l'organisation dont la base doit être
                        interrogée. C'est lui qui décide de la base de
                        connaissances : une réponse ne peut jamais citer le
                        document d'une autre structure.
        provider:       'groq' | 'gemini' | 'local'
        tts:            Générer une réponse vocale.
        tts_out:        Chemin du fichier audio de sortie.
        seed_docs:      Documents en mémoire (si ChromaDB vide).
        seed_filtered:  Docs orientation pour le seed.
        progress:       Callback optionnel appelé à chaque étape avec
                        (step: str, info: dict) — utilisé par le streaming SSE.
        lang_hint:      'wo' | 'fr' pour imposer la langue d'entrée (bouton de
                        la borne). Sans indice, la langue est détectée.
        history:        Mémoire conversationnelle [{role, content}] en français,
                        pour les questions de suite. Le pipeline ne l'écrit pas :
                        l'appelant y ajoute le tour une fois la réponse produite.

    Returns:
        dict {
            trace:        pipeline complet (langues, retrieval, LLM, TTS, latences)
            response_fr:  réponse en français
            response:     réponse dans la langue de l'utilisateur
            response_wo:  réponse en wolof (si entrée wolof)
            question_fr:  question en français (pivot) — à mémoriser par l'appelant
            sources:      documents ayant nourri la réponse (traçabilité RAG)
            audio:        chemin du fichier audio (si tts=True)
        }
    """
    t_total = time.perf_counter()
    trace   = {}

    def _emit(step: str, **info):
        # Toute étape est journalisée, qu'un client écoute ou non le flux SSE.
        # C'est la progression en temps réel : quand une requête semble figée,
        # c'est cette ligne qui dit à quelle étape elle l'est.
        _log.debug("→ %s%s", step,
                   " " + " ".join(f"{c}={v}" for c, v in info.items()
                                  if c not in ("response", "response_fr", "response_wo"))
                   if info else "")
        if progress:
            try:
                progress(step, info)
            except Exception as e:
                # Un client qui ferme sa connexion ne doit pas faire tomber le
                # pipeline, mais l'avaler en silence cacherait une vraie panne.
                _log.debug("callback de progression refusé (%s) — étape %s",
                           type(e).__name__, step)

    # ── 1. Langue d'entrée ────────────────────────────────────────────────
    # L'indice prime sur la détection : quand l'usager a appuyé sur « Wolof »
    # ou « Français », la langue est connue et une détection ne peut que se
    # tromper (phrases courtes, emprunts, code-switching).
    hint = (lang_hint or "").lower()[:2]
    if hint in ("wo", "fr"):
        lang, lang_source = hint, "indice"
    else:
        lang, lang_source = detect_language(unified_text), "détection"
    trace["input_lang"]        = lang
    trace["input_lang_source"] = lang_source
    _emit("detect", lang=lang, source=lang_source)

    # La langue déclarée choisit le moteur STT avant qu'il y ait un texte, et
    # plus rien ne la remet en cause ensuite : un usager qui se trompe de bouton
    # obtient une transcription par le mauvais moteur, puis une réponse dans la
    # mauvaise langue, en silence. On ne peut pas corriger — le mal est fait à la
    # transcription — mais on peut le dire.
    if lang_source == "indice":
        conflit = contredit(lang, unified_text)
        if conflit:
            trace["alerte_langue"] = conflit
            _emit("alerte_langue", **conflit)
            _log.warning("langue déclarée « %s » mais la transcription ressemble "
                         "à « %s » (wolof=%s, français=%s) — texte : %s",
                         conflit["declaree"], conflit["detectee"],
                         conflit["scores"]["wolof"], conflit["scores"]["francais"],
                         " ".join(unified_text.split())[:110])

    # ── 2. Traduction WO→FR si nécessaire ────────────────────────────────
    question_fr = unified_text
    if lang == "wo":
        _emit("translate_in")
        t0 = time.perf_counter()
        question_fr, _ = wolof_to_french(unified_text)
        trace["wolof_to_french"] = {
            "model":   "bilalfaye/nllb-200-distilled-600M-wo-fr-en",
            "result":  question_fr,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    # ── 3. Intention ──────────────────────────────────────────────────────
    # L'embedding de la question sert à la fois au router d'intention et au
    # retrieval : calculé ici une seule fois, la classification ne coûte plus
    # que quelques produits scalaires. En cas d'échec de l'embedder, le router
    # bascule sur ses mots-clés et le retrieval le recalculera.
    t0 = time.perf_counter()
    try:
        q_emb = vectorstore.embed_query(question_fr)
    except Exception:
        q_emb = None
    intent         = detect_intent(question_fr, query_embedding=q_emb)
    trace["intent"] = intent
    trace["intent_latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    _emit("intent", intent=intent)

    # ── 4. RAG ────────────────────────────────────────────────────────────
    _emit("retrieval")
    n_docs = vectorstore.count(organization_id) if organization_id else 0
    trace["organization_id"] = organization_id

    # Une question de suite est complétée par le tour précédent avant recherche.
    query_fr = _contextualize(question_fr, history)
    if query_fr != question_fr:
        trace["retrieval_query"] = query_fr
        q_emb = None                      # l'embedding portait l'ancienne requête

    sources: list[dict] = []
    if n_docs > 0:
        context_fr, retrieval_trace, sources = _rag_pipeline(
            organization_id, query_fr, intent, query_embedding=q_emb
        )
        retrieval_trace["source"] = "chromadb"
    elif seed_docs and settings.SEED_FALLBACK:
        # Repli sur les documents de démonstration, DÉSACTIVÉ par défaut.
        #
        # En multi-tenant ce repli est un piège : une structure fraîchement
        # créée, dont la base est encore vide, répondrait avec les procédures
        # génériques du jeu de démonstration — présentées à l'usager comme les
        # siennes, sans le moindre signe distinctif. Il n'a de sens que sur un
        # poste de démonstration mono-organisation (SEED_FALLBACK=true).
        context_fr, retrieval_trace = _rag_seed(query_fr, intent, seed_docs, seed_filtered)
        retrieval_trace["source"] = "seed"
        _log.warning("repli sur les documents de démonstration "
                     "(organisation=%s, base vide) — SEED_FALLBACK est actif",
                     organization_id)
    else:
        # Base vide : le LLM répondra qu'il ne dispose pas de l'information.
        # Aucun repli, aucune invention, et surtout aucune contamination par
        # les documents d'une autre organisation.
        context_fr      = ""
        retrieval_trace = {"source": "aucun document", "n_candidates": 0}

    trace["retrieval"] = retrieval_trace
    trace["context"]   = context_fr
    trace["n_docs_in_db"] = n_docs

    # ── 5. Génération LLM ─────────────────────────────────────────────────
    _emit("llm", provider=provider)
    t0          = time.perf_counter()
    # plain=True si la réponse sera traduite en wolof (NLLB ne gère pas le markdown)
    response_fr = generate(question_fr, context_fr, provider=provider,
                           plain=(lang == "wo"), history=history)
    trace["llm"] = {
        "provider":   provider,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        "history_messages": len(history or []),
    }

    response = {
        "trace":       trace,
        "response_fr": response_fr,
        "response":    response_fr,
        "question_fr": question_fr,     # version pivot, à mémoriser
        # Traçabilité RAG au premier niveau : quels documents ont nourri cette
        # réponse. Liste vide quand l'organisation n'a aucun document ou
        # qu'aucun passage n'était pertinent — c'est un cas normal, fréquent
        # sur une structure qui vient d'être créée, pas une erreur.
        "sources":     sources,
    }

    # ── QR Code pour les procédures ──────────────────────────────────────────
    if intent == "procedure":
        t0 = time.perf_counter()
        qr_b64 = _generate_qr_code(response_fr)
        if qr_b64:
            response["qr_code"] = qr_b64
        trace["qr_code"] = {
            "generated": qr_b64 is not None,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    # ── 6. Traduction FR→WO ───────────────────────────────────────────────
    if lang == "wo":
        _emit("translate_out")
        t0 = time.perf_counter()
        # Nettoyer le markdown avant traduction (NLLB ne gère pas les balises)
        response_fr_clean = _strip_markdown(response_fr)
        # Puis passer les nombres en chiffres : mesuré, NLLB rend « cinq mille
        # francs » par « junni » (mille) alors que « 5000 francs » traverse
        # intact. Le prompt le demande déjà au LLM, ceci rattrape les oublis.
        response_fr_clean = en_chiffres(response_fr_clean)
        response_wo, _ = french_to_wolof(response_fr_clean)

        # NLLB altère les nombres : mesuré sur 20 questions réelles, 8 réponses
        # en portaient un faux — dont « composez le 3333 » rendu par « 333 »,
        # le numéro d'urgence de l'hôpital. Le français fait autorité et les
        # nombres y sont dans le même ordre : on les réinjecte position par
        # position, mais seulement si les deux textes en portent autant. Sinon
        # l'alignement n'a pas de sens et l'on ne touche à rien.
        response_wo, recal = realigner(response_fr_clean, response_wo)

        response["response_wo"] = response_wo
        response["response"]    = response_wo

        # NLLB réécrit parfois les montants de lui-même, et se trompe : mesuré,
        # « 1000 francs CFA » ressort en « junniy dërëm » (= 5000 F). Quand le
        # modèle fabrique un montant, la valeur d'origine a disparu et rien ne
        # peut la réparer en aval. On ne corrige donc pas : on signale.
        controle = verifier_nombres(response_fr_clean, response_wo)
        trace["french_to_wolof"] = {
            "model":       "bilalfaye/nllb-200-distilled-600M-wo-fr-en",
            "latency_ms":  round((time.perf_counter() - t0) * 1000, 1),
            "nombres":     controle,
            "realignement": recal,
        }
        if recal["corriges"]:
            _log.info(f"réinjectés depuis le français : {recal['corriges']}")
        if not controle["coherent"]:
            _emit("alerte_nombres", **controle)
            # Les extraits sont aplatis : un message de journal qui contient
            # des retours à la ligne casse un `grep` et brouille la lecture.
            _plat = lambda t: " ".join((t or "").split())[:110]
            _log.warning("écart de nombres FR/WO — suspects=%s ajoutés=%s | "
                         "fr: %s | wo: %s", controle["suspects"], controle["ajoutes"],
                         _plat(response_fr_clean), _plat(response_wo))

    # ── 7. TTS (optionnel) ────────────────────────────────────────────────
    if tts:
        # La réponse part avec l'annonce de la synthèse : `result` n'arrive
        # qu'une fois l'audio produit, et la borne se mettrait sinon à parler
        # une minute avant d'afficher ce qu'elle dit.
        _emit("tts",
              response    = response.get("response", ""),
              response_fr = response.get("response_fr", ""),
              response_wo = response.get("response_wo"),
              lang        = lang)
        from tts_Ooleil.tts import synthesize, source as tts_source
        t0           = time.perf_counter()
        text_for_tts = response.get("response_wo", response_fr)
        # Oolel-Voices ne lit correctement que du wolof en toutes lettres :
        # chiffres, horaires, pourcentages et sigles se dégradent, et certains
        # (« APIX », « 8:00 ») détruisent l'énoncé entier. Le texte est donc
        # préparé pour la synthèse — l'affichage, lui, garde ses chiffres.
        bilan_tts = {}
        if lang == "wo":
            text_for_tts, bilan_tts = pour_synthese(text_for_tts)

        # La synthèse est le poste le plus lent du pipeline : sur une réponse
        # longue, 123 s pour 61 s d'audio. Chaque morceau est donc annoncé dès
        # qu'il est prêt (le premier à 23 s), pour que le client commence à
        # lire au lieu d'attendre l'énoncé complet.
        morceaux = []

        def _sur_morceau(index: int, total: int, chemin: str) -> None:
            morceaux.append(chemin)
            _emit("tts_chunk", index=index, total=total, audio=chemin,
                  latency_ms=round((time.perf_counter() - t0) * 1000, 1))

        synthesize(text_for_tts, tts_out, on_chunk=_sur_morceau)
        response["audio"] = tts_out
        response["audio_chunks"] = morceaux
        trace["tts"] = {
            "engine":     tts_source(),
            "texte_synthetise": text_for_tts,
            "n_morceaux": len(morceaux),
            **bilan_tts,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    # ── 8. Latence totale ─────────────────────────────────────────────────
    trace["total_latency_ms"] = round((time.perf_counter() - t_total) * 1000, 1)

    # La répartition en une ligne. C'est elle qu'on lit quand « la borne est
    # lente » : elle désigne le poste fautif sans avoir à ouvrir la trace JSON.
    postes = [
        ("wo→fr",  trace.get("wolof_to_french", {}).get("latency_ms")),
        ("intent", trace.get("intent_latency_ms")),
        ("recher", trace.get("retrieval", {}).get("latency_total_ms")),
        ("llm",    trace.get("llm", {}).get("latency_ms")),
        ("fr→wo",  trace.get("french_to_wolof", {}).get("latency_ms")),
        ("tts",    trace.get("tts", {}).get("latency_ms")),
    ]
    detail = "  ".join(f"{nom}={ms:.0f}" for nom, ms in postes if ms)
    _log.info("%s · %d car. · total=%.0f ms   %s",
              trace.get("intent", "?"), len(response.get("response", "")),
              trace["total_latency_ms"], detail)

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
