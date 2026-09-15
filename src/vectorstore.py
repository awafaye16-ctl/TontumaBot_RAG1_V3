"""Vectorstore V3.1 — une base vectorielle par organisation.

Isolation multi-tenant
----------------------
Chaque organisation possède sa PROPRE base ChromaDB, dans son propre
répertoire :

    data/chroma/{organization_id}/
        ├── chroma.sqlite3
        └── meta.json     {organization_id, embed_model, cree_le, maj_le, ...}

L'isolation est donc structurelle, et c'est un choix délibéré contre le filtre
de métadonnées sur index unique : la recherche hybride ci-dessous ne passe PAS
par le moteur de requête de ChromaDB — elle travaille sur un cache mémoire
(index BM25 + matrice d'embeddings), où un `where` n'aurait aucun effet. Il
aurait fallu réimplémenter le filtrage sur chaque chemin de recherche, et un
seul oubli aurait produit une fuite silencieuse d'un client vers un autre.
Ici, le pire défaut possible est une base vide ou une exception — jamais la
réponse d'une mairie citant le dossier d'un hôpital.

Toutes les fonctions publiques prennent l'organisation en premier argument.

Pipeline de recherche (inchangé, mais borné à une organisation) :
  1. Hybrid search   BM25 (lexical) + vectoriel (sémantique)  → top-K candidats
  2. MMR             Maximum Marginal Relevance               → diversité
  3. Reranker        cross-encoder                            → tri final

Toutes les fonctions de recherche retournent des tuples
  (chunk_id, texte, score, metadata)
pour que le pipeline puisse tracer chaque étape.
"""
import hashlib
import json
import os
import shutil
import sys
import threading
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import settings  # noqa: E402

import chromadb
import numpy as np

from journal import journal

_log = journal("vectorstore")

EMBED_MODEL = settings.EMBED_MODEL
COLLECTION  = "documents"          # même nom partout : l'isolation vient du répertoire
DB_ROOT     = settings.CHROMA_ROOT

_embedder = None


# =============================================================================
#  Erreurs
# =============================================================================

class OrganisationInvalide(ValueError):
    """`organization_id` absent ou qui n'est pas un UUID."""


class IndexIncompatible(RuntimeError):
    """La base a été construite avec un autre modèle d'embedding.

    Levée au lieu de purger : en V3.0 un changement d'`EMBED_MODEL` effaçait
    silencieusement la collection. Avec une base par client, le même réflexe
    aurait effacé les données de toutes les structures au premier démarrage
    suivant la modification d'une variable d'environnement.
    """


# =============================================================================
#  Identifiant d'organisation
# =============================================================================

def valider_organisation(organization_id: Optional[str]) -> str:
    """Valide l'identifiant et retourne sa forme canonique (minuscules, tirets).

    Le retour de cette fonction — et JAMAIS l'entrée brute — est ce qui sert de
    segment de chemin. Le service n'ayant aucune authentification, un
    `organization_id` arrive tel quel depuis le réseau : concaténer l'entrée à
    `data/chroma/` laisserait passer `../../` et donnerait un accès en écriture
    arbitraire au disque. Passer par `uuid.UUID` élimine la question, quelle que
    soit la forme envoyée.
    """
    if not organization_id or not str(organization_id).strip():
        raise OrganisationInvalide("organization_id requis")
    try:
        return str(uuid.UUID(str(organization_id).strip()))
    except (ValueError, AttributeError, TypeError):
        raise OrganisationInvalide(
            f"organization_id invalide (UUID attendu) : {organization_id!r}"
        ) from None


def chemin_organisation(organization_id: str) -> str:
    return os.path.join(DB_ROOT, valider_organisation(organization_id))


# =============================================================================
#  Embedding — partagé par TOUTES les organisations
# =============================================================================
#  Seules les DONNÉES sont cloisonnées. Le modèle d'embedding pèse plusieurs
#  centaines de mégaoctets : le charger par organisation ferait exploser la RAM
#  au deuxième client. Il reste donc un singleton de processus, comme NLLB, le
#  reranker, le STT et le TTS.

def get_embedder() -> "SentenceTransformer":
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer  # import lazy
        import device as _dev
        dev = _dev.resolve("embedder")
        _log.info(f"Chargement embedder ({EMBED_MODEL}) sur {dev.upper()}...")
        # Sans `device`, sentence-transformers refait sa propre détection : on
        # lui impose celle du pipeline pour que tout partage le même backend.
        _embedder = SentenceTransformer(EMBED_MODEL, device=dev)
    return _embedder


def _embed(texts: list[str]) -> list[list[float]]:
    return get_embedder().encode(texts, normalize_embeddings=True).tolist()


def _embed_single(text: str) -> list[float]:
    return _embed([text])[0]


def embed_query(text: str) -> list[float]:
    """Embedding normalisé de la requête (à calculer une seule fois par question)."""
    return _embed_single(text)


# =============================================================================
#  Registre des bases ouvertes — cache LRU
# =============================================================================
#  Chaque organisation ouverte coûte une connexion SQLite, un index HNSW, un
#  index BM25 et sa matrice d'embeddings (~5 Mo pour 1 000 fragments en 384
#  dimensions). Sans plafond, le dictionnaire grossirait jusqu'à la dernière
#  organisation servie depuis le démarrage.
#
#  Le pipeline tourne dans un executor : plusieurs requêtes touchent ce registre
#  en même temps. `_verrou_registre` protège la structure (ouverture, éviction) ;
#  chaque base a en plus son propre verrou pour la construction de son corpus,
#  qui prend quelques secondes et ne doit pas bloquer les autres organisations.

class _Base:
    """Une organisation ouverte : son client, sa collection, son cache de corpus."""

    __slots__ = ("organisation", "chemin", "client", "collection", "corpus", "verrou")

    def __init__(self, organisation: str, chemin: str, client, collection):
        self.organisation = organisation
        self.chemin       = chemin
        self.client       = client
        self.collection   = collection
        self.corpus       = None
        self.verrou       = threading.RLock()

    def fermer(self) -> None:
        try:
            self.client.close()
        except Exception as e:  # noqa: BLE001
            # Fermer proprement un client individuel reste mal outillé côté
            # ChromaDB. Un échec ici ne doit pas faire tomber la requête : au
            # pire la connexion sera libérée à l'arrêt du processus.
            _log.debug("fermeture de %s imparfaite (%s)", self.organisation, e)


_bases: "OrderedDict[str, _Base]" = OrderedDict()
_verrou_registre = threading.RLock()


def _meta_path(chemin: str) -> str:
    return os.path.join(chemin, "meta.json")


def _lire_meta(chemin: str) -> dict:
    try:
        with open(_meta_path(chemin), "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _ecrire_meta(chemin: str, **champs) -> None:
    """Écrit meta.json — le répertoire d'une organisation reste auto-descriptif.

    Ce n'est pas décoratif : c'est lui qui dit quel modèle d'embedding a produit
    l'index (donc si une réindexation est due), et c'est lui qui rend crédible la
    livraison du répertoire à une structure qui passerait sur site.
    """
    meta = _lire_meta(chemin)
    meta.update(champs)
    meta["maj_le"] = datetime.now(timezone.utc).isoformat()
    meta.setdefault("cree_le", meta["maj_le"])
    meta["embed_model"] = EMBED_MODEL
    try:
        with open(_meta_path(chemin), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except OSError as e:
        _log.warning("meta.json non écrit (%s) : %s", chemin, e)


def _ouvrir(organisation: str) -> _Base:
    """Ouvre (ou crée) la base d'une organisation."""
    chemin = os.path.join(DB_ROOT, organisation)
    nouvelle = not os.path.isdir(chemin)
    os.makedirs(chemin, exist_ok=True)

    # Une base construite avec un autre modèle d'embedding n'est pas réparable
    # à chaud : les vecteurs stockés ne sont pas comparables aux nouveaux. On
    # refuse de l'ouvrir plutôt que de rendre des résultats faux — ou de purger.
    meta = _lire_meta(chemin)
    modele_precedent = meta.get("embed_model")
    if modele_precedent and modele_precedent != EMBED_MODEL:
        raise IndexIncompatible(
            f"organisation {organisation} : index construit avec "
            f"'{modele_precedent}', or EMBED_MODEL vaut '{EMBED_MODEL}'. "
            f"Réindexez cette organisation, ou rétablissez le modèle précédent. "
            f"Aucune donnée n'a été touchée."
        )

    client     = chromadb.PersistentClient(path=chemin)
    collection = client.get_or_create_collection(
        COLLECTION,
        metadata={"hnsw:space": "cosine", "embed_model": EMBED_MODEL},
    )
    if nouvelle:
        _log.info("organisation %s : base créée", organisation)
    _ecrire_meta(chemin, organization_id=organisation)
    return _Base(organisation, chemin, client, collection)


def existe(organization_id: str) -> bool:
    """L'organisation a-t-elle déjà une base sur le disque ?"""
    return os.path.isdir(os.path.join(DB_ROOT, valider_organisation(organization_id)))


def _base(organization_id: str, creer: bool = True) -> Optional[_Base]:
    """Retourne la base de l'organisation, en l'ouvrant si besoin (LRU).

    `creer=False` retourne None au lieu de créer la base : une LECTURE ne doit
    pas laisser de trace sur le disque. Sans ça, interroger une organisation
    inexistante — une faute de frappe, une sonde, un balayage — sèmerait des
    répertoires vides que plus rien ne distingue d'un client réel.
    """
    org = valider_organisation(organization_id)
    with _verrou_registre:
        base = _bases.get(org)
        if base is not None:
            _bases.move_to_end(org)
            return base

        if not creer and not os.path.isdir(os.path.join(DB_ROOT, org)):
            return None

        base = _ouvrir(org)
        _bases[org] = base

        # Éviction des bases les moins récemment utilisées
        while len(_bases) > max(1, settings.MAX_ORGANISATIONS_EN_CACHE):
            _, ancienne = _bases.popitem(last=False)
            _log.debug("éviction du cache : organisation %s", ancienne.organisation)
            ancienne.fermer()
        return base


def get_collection(organization_id: str, creer: bool = True):
    base = _base(organization_id, creer=creer)
    return base.collection if base is not None else None


def organisations_chargees() -> int:
    with _verrou_registre:
        return len(_bases)


def organisations() -> list[dict]:
    """Liste les organisations présentes sur le disque, d'après leurs meta.json."""
    if not os.path.isdir(DB_ROOT):
        return []
    out = []
    for nom in sorted(os.listdir(DB_ROOT)):
        chemin = os.path.join(DB_ROOT, nom)
        if not os.path.isdir(chemin):
            continue
        try:
            org = valider_organisation(nom)
        except OrganisationInvalide:
            continue          # répertoire étranger (ex. base V3.0 non migrée)
        meta = _lire_meta(chemin)
        out.append({
            "organization_id": org,
            "cree_le":     meta.get("cree_le"),
            "maj_le":      meta.get("maj_le"),
            "embed_model": meta.get("embed_model"),
            "n_documents": meta.get("n_documents"),
            "n_chunks":    meta.get("n_chunks"),
        })
    return out


# =============================================================================
#  Cache du corpus (index BM25 + embeddings) — par organisation
# =============================================================================

def _get_corpus(organization_id: str) -> dict:
    """Charge le corpus d'UNE organisation et construit son index BM25.

    Réutilisé pour toutes ses requêtes jusqu'à la prochaine écriture
    (add/delete/clear), qui invalide le cache via `_invalidate_corpus`.

    L'index BM25 est construit sur les seuls documents de cette organisation :
    ses statistiques de fréquence (IDF) sont donc calculées sur son propre
    corpus. Un index partagé aurait dilué le pouvoir discriminant d'un terme
    rare chez un client mais courant chez un autre — une dégradation de
    pertinence invisible en test, qui grandit avec le nombre de clients.
    """
    base = _base(organization_id, creer=False)
    if base is None:
        # Organisation sans base : corpus vide, et surtout rien de créé.
        return {"ids": [], "texts": [], "metas": [], "embeddings": None,
                "bm25": None, "id_to_pos": {}}
    with base.verrou:
        if base.corpus is not None:
            return base.corpus

        data  = base.collection.get(include=["documents", "metadatas", "embeddings"])
        ids   = data["ids"]
        texts = data["documents"]
        metas = data["metadatas"]

        embeddings = data.get("embeddings")
        embeddings = np.asarray(embeddings) if embeddings is not None and len(embeddings) else None

        bm25 = None
        if texts:
            from retrieval.hybrid import HybridRetriever
            bm25 = HybridRetriever(texts)

        base.corpus = {
            "ids":        ids,
            "texts":      texts,
            "metas":      metas,
            "embeddings": embeddings,
            "bm25":       bm25,
            "id_to_pos":  {cid: i for i, cid in enumerate(ids)},
        }
        return base.corpus


def _invalidate_corpus(organization_id: str) -> None:
    with _verrou_registre:
        base = _bases.get(valider_organisation(organization_id))
    if base is not None:
        with base.verrou:
            base.corpus = None


def prechauffer(organization_id: str) -> int:
    """Construit d'avance le corpus d'une organisation.

    Le premier appel d'une organisation paie l'ouverture du client, le
    chargement des vecteurs et la construction de l'index BM25 — une à trois
    secondes, visibles sur une borne tactile. Appelé au démarrage pour les
    organisations déjà présentes sur le disque.
    """
    return len(_get_corpus(organization_id)["ids"])


# =============================================================================
#  Écriture
# =============================================================================

def add_documents(organization_id: str, chunks: list[str], metadatas: list[dict]) -> int:
    """Ajoute (ou met à jour) des fragments dans la base de l'organisation.

    L'ID d'un fragment est un hash MD5 de (texte + metadata) : ingérer deux fois
    le même contenu ne crée pas de doublons.

    ⚠ Cette idempotence ne suffit PAS à remplacer un document dont le contenu a
    changé : le texte ayant changé, les nouveaux fragments portent de nouveaux
    identifiants et l'`upsert` les ajoute À CÔTÉ des anciens. Le remplacement
    propre passe par `ingestion.ingest_*`, qui supprime d'abord par
    `document_id`.
    """
    org = valider_organisation(organization_id)
    if not chunks:
        return 0
    base = _base(org)          # écriture : la base est créée si besoin
    ids  = [
        "doc-" + hashlib.md5((c + "|" + str(m)).encode()).hexdigest()[:16]
        for c, m in zip(chunks, metadatas)
    ]
    embeddings = _embed(chunks)
    base.collection.upsert(ids=ids, documents=chunks,
                           embeddings=embeddings, metadatas=metadatas)
    _invalidate_corpus(org)
    _maj_compteurs(org)
    return len(chunks)


def _maj_compteurs(organization_id: str) -> None:
    """Reporte les comptages dans meta.json (répertoire auto-descriptif)."""
    org  = valider_organisation(organization_id)
    base = _base(org, creer=False)
    if base is None:
        return
    try:
        _ecrire_meta(base.chemin,
                     organization_id=org,
                     n_chunks=base.collection.count(),
                     n_documents=len(all_documents(org)))
    except Exception as e:  # noqa: BLE001
        _log.debug("compteurs non mis à jour pour %s : %s", org, e)


# =============================================================================
#  Recherche vectorielle pure
# =============================================================================

def vector_search(
    organization_id: str,
    query: str,
    k: int = 10,
    filter_meta: Optional[dict] = None,
) -> list[tuple[str, str, float, dict]]:
    """Recherche vectorielle cosinus, bornée à l'organisation.

    Returns:
        Liste de (chunk_id, texte, score_similarité, metadata), triée par score desc.
    """
    coll = get_collection(organization_id, creer=False)
    if coll is None or coll.count() == 0:
        return []

    kwargs = {"query_embeddings": [_embed_single(query)], "n_results": k,
              "include": ["documents", "metadatas", "distances"]}
    if filter_meta:
        kwargs["where"] = filter_meta

    res = coll.query(**kwargs)
    out = []
    for doc, meta, dist, cid in zip(
        res["documents"][0], res["metadatas"][0],
        res["distances"][0],  res["ids"][0],
    ):
        out.append((cid, doc, 1.0 - dist, meta))  # distance cosinus → similarité
    return out


# =============================================================================
#  Recherche hybride BM25 + vectorielle
# =============================================================================

def hybrid_search(
    organization_id: str,
    query: str,
    k: int   = 10,
    alpha: float = 0.4,
    query_embedding: Optional[list[float]] = None,
) -> list[tuple[str, str, float, dict]]:
    """Fusionne BM25 (lexical) et vectoriel (sémantique), dans une seule organisation.

    score_final = alpha * score_BM25 + (1 - alpha) * score_vectoriel

    alpha = 0.4 → légère préférence au sémantique (meilleur sur les questions
    administratives paraphrasées).

    L'index BM25 et les embeddings du corpus sont mis en cache (`_get_corpus`)
    par organisation. L'embedding de la requête peut être fourni via
    `query_embedding` pour éviter de le recalculer.

    Returns:
        Liste de (chunk_id, texte, score_fusionné, metadata), top-K.
    """
    corpus = _get_corpus(organization_id)
    ids, texts, metas = corpus["ids"], corpus["texts"], corpus["metas"]
    if not ids:
        return []

    # ── BM25 (index caché, propre à l'organisation) ───────────────────────
    bm25_scores = dict(corpus["bm25"]._bm25_search(query, k=k * 3))

    # ── Vectoriel (embeddings cachés, similarité cosinus = produit scalaire
    #    car les vecteurs sont normalisés) ──────────────────────────────────
    vec_scores = {}
    embeddings = corpus["embeddings"]
    if embeddings is not None:
        if query_embedding is None:
            query_embedding = _embed_single(query)
        sims  = embeddings @ np.asarray(query_embedding)
        top_n = min(k * 3, len(sims))
        for idx in np.argpartition(sims, -top_n)[-top_n:]:
            vec_scores[int(idx)] = float(sims[idx])
    else:
        # Corpus sans embeddings en cache → repli sur la requête ChromaDB
        id_to_pos = corpus["id_to_pos"]
        for cid, _, s, _ in vector_search(organization_id, query, k=k * 3):
            if cid in id_to_pos:
                vec_scores[id_to_pos[cid]] = s

    # ── Fusion ────────────────────────────────────────────────────────────
    results = []
    for idx in set(bm25_scores) | set(vec_scores):
        s_b = bm25_scores.get(idx, 0.0)
        s_v = vec_scores.get(idx, 0.0)
        results.append((ids[idx], texts[idx], alpha * s_b + (1 - alpha) * s_v, metas[idx]))

    results.sort(key=lambda x: x[2], reverse=True)
    return results[:k]


# =============================================================================
#  MMR — Maximum Marginal Relevance
# =============================================================================

def mmr_from_candidates(
    organization_id: str,
    candidates: list[tuple[str, str, float, dict]],
    query_embedding: list[float],
    k: int = 5,
    lambda_mmr: float = 0.6,
) -> list[tuple[str, str, float, dict]]:
    """Applique le MMR sur des candidats DÉJÀ récupérés (pas de nouvelle recherche).

    MMR(d) = lambda * sim(query, d) - (1 - lambda) * max_sim(d, already_selected)

    Les vecteurs des candidats sont récupérés depuis le cache d'embeddings de
    l'organisation ; on ne les ré-encode donc pas.

    Returns:
        Liste de (chunk_id, texte, score_mmr, metadata), k éléments distincts.
    """
    if not candidates:
        return []

    # ── Déduplication par texte exact avant MMR ───────────────────────────
    seen_texts: set[str] = set()
    deduped: list = []
    for item in candidates:
        txt = item[1].strip()
        if txt not in seen_texts:
            seen_texts.add(txt)
            deduped.append(item)
    candidates = deduped

    q_vec     = np.asarray(query_embedding)
    corpus    = _get_corpus(organization_id)
    id_to_pos = corpus["id_to_pos"]
    emb       = corpus["embeddings"]

    # Vecteurs des candidats : depuis le cache si possible, sinon ré-encodage
    if emb is not None and all(c[0] in id_to_pos for c in candidates):
        cand_vecs = np.stack([emb[id_to_pos[c[0]]] for c in candidates])
    else:
        cand_vecs = get_embedder().encode(
            [c[1] for c in candidates], normalize_embeddings=True
        )

    selected_idx  : list[int]   = []
    selected_vecs : list[np.ndarray] = []
    remaining_idx : list[int]   = list(range(len(candidates)))

    for _ in range(min(k, len(candidates))):
        best_idx   = -1
        best_score = -float("inf")

        for i in remaining_idx:
            relevance = float(np.dot(q_vec, cand_vecs[i]))
            if selected_vecs:
                redundancy = max(
                    float(np.dot(cand_vecs[i], sv)) for sv in selected_vecs
                )
            else:
                redundancy = 0.0

            mmr_score = lambda_mmr * relevance - (1 - lambda_mmr) * redundancy
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx   = i

        selected_idx.append(best_idx)
        selected_vecs.append(cand_vecs[best_idx])
        remaining_idx.remove(best_idx)

    return [
        (candidates[i][0], candidates[i][1], float(np.dot(q_vec, cand_vecs[i])), candidates[i][3])
        for i in selected_idx
    ]


def mmr_search(
    organization_id: str,
    query: str,
    k: int      = 5,
    fetch_k: int = 20,
    lambda_mmr: float = 0.6,
    query_embedding: Optional[list[float]] = None,
) -> list[tuple[str, str, float, dict]]:
    """Récupère fetch_k candidats par hybride puis sélectionne k fragments par MMR."""
    if query_embedding is None:
        query_embedding = _embed_single(query)
    candidates = hybrid_search(organization_id, query, k=fetch_k,
                               query_embedding=query_embedding)
    return mmr_from_candidates(organization_id, candidates, query_embedding,
                               k=k, lambda_mmr=lambda_mmr)


# =============================================================================
#  Recherche filtrée par métadonnée (pour l'intention "orientation")
# =============================================================================

def filtered_search(
    organization_id: str,
    query: str,
    filter_meta: dict,
    k: int = 5,
) -> list[tuple[str, str, float, dict]]:
    """Restreint la recherche aux fragments ayant des métadonnées spécifiques."""
    return vector_search(organization_id, query, k=k, filter_meta=filter_meta)


# =============================================================================
#  Gestion des documents
# =============================================================================

def all_documents(organization_id: str) -> list[dict]:
    """Documents indexés de l'organisation (dédoublonnés par document_id).

    Organisation sans base : liste vide, et aucune base créée au passage.
    """
    coll = get_collection(organization_id, creer=False)
    if coll is None:
        return []
    data = coll.get(include=["documents", "metadatas"])
    docs: dict = {}
    for doc, meta in zip(data["documents"], data["metadatas"]):
        did = meta.get("document_id", "?")
        if did not in docs:
            docs[did] = {
                "id":       did,
                "title":    meta.get("title", did),
                "source":   meta.get("source", ""),
                "category": meta.get("category") or None,
                "chunks":   0,
                "added":    meta.get("added", ""),
            }
        docs[did]["chunks"] += 1
    return list(docs.values())


def document_exists(organization_id: str, document_id: str) -> bool:
    coll = get_collection(organization_id, creer=False)
    if coll is None:
        return False
    return bool(coll.get(where={"document_id": document_id}, limit=1)["ids"])


def delete_document(organization_id: str, document_id: str) -> int:
    org  = valider_organisation(organization_id)
    coll = get_collection(org, creer=False)
    if coll is None:
        return 0
    data = coll.get(where={"document_id": document_id})
    ids  = data["ids"]
    if ids:
        coll.delete(ids=ids)
        _invalidate_corpus(org)
        _maj_compteurs(org)
    return len(ids)


def clear_all(organization_id: str) -> int:
    """Efface tous les documents de CETTE organisation. Les autres sont intactes."""
    org  = valider_organisation(organization_id)
    coll = get_collection(org, creer=False)
    if coll is None:
        return 0
    n    = coll.count()
    if n:
        coll.delete(ids=coll.get()["ids"])
        _invalidate_corpus(org)
        _maj_compteurs(org)
    return n


def delete_organization(organization_id: str) -> dict:
    """Supprime la base d'une organisation, répertoire compris. Irréversible.

    À appeler à la résiliation d'un client : c'est la garantie d'effacement qui
    se démontre — il ne reste rien sur le disque, pas des lignes filtrées.
    """
    org = valider_organisation(organization_id)
    chemin = os.path.join(DB_ROOT, org)
    if not os.path.isdir(chemin):
        return {"existed": False, "deleted_chunks": 0, "deleted_documents": 0}

    try:
        n_chunks    = count(org)
        n_documents = len(all_documents(org))
    except Exception:  # noqa: BLE001
        n_chunks, n_documents = 0, 0

    with _verrou_registre:
        base = _bases.pop(org, None)
    if base is not None:
        base.fermer()

    shutil.rmtree(chemin, ignore_errors=True)
    _log.warning("organisation %s supprimée (%d documents, %d fragments)",
                 org, n_documents, n_chunks)
    return {"existed": True, "deleted_chunks": n_chunks, "deleted_documents": n_documents}


def count(organization_id: str) -> int:
    coll = get_collection(organization_id, creer=False)
    return coll.count() if coll is not None else 0
