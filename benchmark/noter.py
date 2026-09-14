"""Note un fichier de résultats bruts et produit le rapport.

Séparé de l'exécution à dessein : un passage coûte des minutes et des appels
payants, une grille de notation se corrige souvent. On renote sans reproduire.

Trois familles de mesures, d'inégale solidité — le rapport le dit :

  MÉCANIQUE   latences, longueurs, abstentions, alertes. Rien à interpréter.
  EXACTE      rappel du retrieval : les identifiants de passages sont comparés,
              pas des sacs de mots. Vrai ou faux, sans jugement.
  JUGÉE       couverture des faits et contradictions. Un modèle décide, avec
              les biais d'un modèle — d'où le contrôle lexical en parallèle.

La couverture est séparée en essentiels et complémentaires : le système a
consigne de répondre en trois phrases, la référence décrit une réponse complète.
Confondre les deux reviendrait à punir la brièveté qu'on a demandée.

Usage :
    venv/bin/python benchmark/noter.py brut_20260914_1106_p3.json
    venv/bin/python benchmark/noter.py --tous        # tous les fichiers bruts
"""
import argparse
import json
import os
import re
import statistics as st
import sys
import unicodedata
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "src"))
os.chdir(RACINE)

DOSSIER    = RACINE / "benchmark" / "resultats"
REFERENCES = RACINE / "benchmark" / "references.json"
MAX_JETONS = int(os.getenv("REFERENCE_MAX_TOKENS", "4000"))

CONSIGNE_JUGE = """Tu notes la réponse d'un assistant hospitalier.

On te donne une question, la réponse produite, et une liste numérotée de faits
que la réponse idéale contiendrait. Pour chaque fait, dis s'il est porté par la
réponse.

Rends un objet JSON :

  "presents"    : liste des numéros de faits réellement portés par la réponse.
  "contredit"   : liste des numéros de faits que la réponse CONTREDIT.
  "invente"     : true si la réponse affirme un fait précis (lieu, montant,
                  délai, numéro) qui n'apparaît dans aucun fait de la liste.
  "detail"      : une phrase sur ce qui est contredit ou inventé, sinon "".

Règles :
- Une reformulation compte : « apportez votre carte d'identité » porte le fait
  « le patient doit présenter sa pièce d'identité ».
- Une formulation plus brève compte, tant que l'information y est.
- Une allusion vague ne compte pas : « apportez les documents nécessaires » ne
  porte PAS un fait qui énumère des pièces précises.
- Dans le doute, considère le fait comme absent. Un juge complaisant ne mesure
  rien.
"""

# Formules d'abstention du prompt système, plus les variantes observées.
_ABSTENTIONS = [
    "pas trouvé cette information", "pas trouve cette information",
    "base documentaire", "aucune information", "je ne sais pas",
    "ne figure pas", "pas mentionné", "pas mentionne",
]


def _sans_accents(t: str) -> str:
    t = unicodedata.normalize("NFKD", (t or "").lower())
    return "".join(c for c in t if not unicodedata.combining(c))


def est_abstention(reponse: str) -> bool:
    plat = _sans_accents(reponse)
    return any(_sans_accents(m) in plat for m in _ABSTENTIONS)


def indices_presents(reponse: str, indices: list[str]) -> float:
    """Part des indices lexicaux retrouvés — contrôle parallèle au juge.

    Volontairement naïf : il ne sert pas à noter mais à repérer un juge qui
    dérive. Un écart durable entre les deux colonnes doit alerter.
    """
    if not indices:
        return 0.0
    plat = _sans_accents(reponse)
    trouves = sum(1 for i in indices if _sans_accents(i) in plat)
    return trouves / len(indices)


def juger(client, modele: str, question: str, reponse: str, faits: list[dict]) -> dict:
    liste = "\n".join(f"{i + 1}. [{f['importance']}] {f['fait']}"
                      for i, f in enumerate(faits))
    r = client.chat.completions.create(
        model=modele,
        messages=[
            {"role": "system", "content": CONSIGNE_JUGE},
            {"role": "user", "content":
                f"Question : {question}\n\nRéponse produite :\n{reponse}\n\nFaits attendus :\n{liste}"},
        ],
        temperature=0,
        max_tokens=MAX_JETONS,
        response_format={"type": "json_object"},
    )
    d = json.loads(r.choices[0].message.content or "{}")
    valides = lambda cle: {n for n in d.get(cle, [])
                           if isinstance(n, int) and 1 <= n <= len(faits)}
    return {"presents": valides("presents"), "contredit": valides("contredit"),
            "invente": bool(d.get("invente")), "detail": (d.get("detail") or "").strip()}


def percentiles(valeurs: list[float]) -> dict:
    propres = sorted(v for v in valeurs if isinstance(v, (int, float)))
    if not propres:
        return {}
    pick = lambda p: propres[min(len(propres) - 1, int(len(propres) * p))]
    return {"n": len(propres), "p50": round(pick(0.50), 1),
            "p95": round(pick(0.95), 1), "max": round(propres[-1], 1)}


def noter(chemin: Path, refs: dict, client, modele: str) -> dict:
    brut      = json.loads(chemin.read_text())
    resultats = [r for r in brut["resultats"] if not r.get("echec")]
    echecs    = [r for r in brut["resultats"] if r.get("echec")]

    par_question, jugements = [], {}
    for r in resultats:
        ref = refs.get(r["id"])
        if not ref:
            continue
        abstenu = est_abstention(r["response_fr"])

        # ── Rappel du retrieval : comparaison d'identifiants, pas de mots ──
        attendus = set(ref.get("chunks_ids") or [])
        obtenus  = r.get("chunks") or []
        rappel = len(attendus & set(obtenus)) / len(attendus) if attendus else None
        rang = next((i + 1 for i, c in enumerate(obtenus) if c in attendus), None)

        entree = {
            "id": r["id"], "repondable": ref["repondable"], "abstenu": abstenu,
            "rappel": rappel, "premier_rang_utile": rang,
            "n_car": len(r["response_fr"]), "duree_s": r["duree_s"],
        }

        faits = ref.get("faits") or []
        if ref["repondable"] and faits and not abstenu:
            cle = (r["id"], r["response_fr"])
            if cle not in jugements:
                jugements[cle] = juger(client, modele, ref["question_fr"],
                                       r["response_fr"], faits)
            j = jugements[cle]
            ess = [i for i, f in enumerate(faits, 1) if f["importance"] == "essentiel"]
            cmp_ = [i for i, f in enumerate(faits, 1) if f["importance"] != "essentiel"]
            entree.update({
                "essentiels":      len(set(ess) & j["presents"]) / len(ess) if ess else None,
                "complementaires": len(set(cmp_) & j["presents"]) / len(cmp_) if cmp_ else None,
                "contredit":       len(j["contredit"]),
                "invente":         j["invente"],
                "detail":          j["detail"],
                "indices":         st.mean([indices_presents(r["response_fr"], f["indices"])
                                            for f in faits]) if faits else None,
            })
        par_question.append(entree)

    moy = lambda cle: (round(st.mean([e[cle] for e in par_question
                                      if e.get(cle) is not None]), 3)
                       if any(e.get(cle) is not None for e in par_question) else None)
    repondables = [e for e in par_question if e["repondable"]]
    absentes    = [e for e in par_question if not e["repondable"]]

    etapes = ["wo_fr", "intent", "retrieval", "rerank", "llm", "fr_wo", "tts", "total"]
    return {
        "fichier":        chemin.name,
        "configuration":  brut["configuration"],
        "integrite":      {"complet": brut.get("complet", True),
                           "attendu": brut.get("attendu"),
                           "n_ok":    len(resultats)},
        "n_executions":   len(resultats),
        "n_echecs":       len(echecs),
        "latences_ms":    {e: percentiles([r["latences_ms"].get(e) for r in resultats])
                           for e in etapes},
        "longueur_car":   percentiles([len(r["response_fr"]) for r in resultats]),
        "retrieval":      {"rappel_moyen": moy("rappel"),
                           "rang_median_utile": st.median(
                               [e["premier_rang_utile"] for e in par_question
                                if e["premier_rang_utile"]]) if any(
                               e["premier_rang_utile"] for e in par_question) else None,
                           "jamais_trouve": sum(1 for e in par_question
                                                if e["rappel"] == 0)},
        "abstention":     {"attendue_et_faite": sum(1 for e in absentes if e["abstenu"]),
                           "attendue":          len(absentes),
                           "a_tort":            sum(1 for e in repondables if e["abstenu"])},
        "couverture":     {"essentiels": moy("essentiels"),
                           "complementaires": moy("complementaires"),
                           "controle_lexical": moy("indices")},
        "fidelite":       {"contradictions": sum(e.get("contredit", 0) for e in par_question),
                           "inventions":     sum(1 for e in par_question if e.get("invente"))},
        "alertes_langue": sum(1 for r in resultats if r.get("alerte_langue")),
        "par_question":   par_question,
    }


def afficher(rapport: dict) -> None:
    cfg = rapport["configuration"]
    brut_complet = rapport.get("integrite", {})
    print("═" * 74)
    if not brut_complet.get("complet", True):
        print(f"  ⚠️  PASSAGE INCOMPLET — {brut_complet.get('n_ok', '?')} exécutions sur "
              f"{brut_complet.get('attendu', '?')} attendues.")
        print(f"      Les percentiles portent sur ce qui a abouti, pas sur le jeu entier.")
    print(f"  {rapport['fichier']}")
    print(f"  commit {cfg['commit']}{'' if cfg['arbre_propre'] else ' (non commité)'} · "
          f"{cfg['llm_max_phrases']} phrases · {cfg['n_chunks_indexes']} chunks · "
          f"{cfg['backends']['auto']} · {cfg['machine']}")
    print("═" * 74)

    print("\n  MÉCANIQUE — rien à interpréter")
    print(f"    {'étape':<12} {'p50':>9} {'p95':>9} {'max':>9}")
    for etape, v in rapport["latences_ms"].items():
        if v:
            print(f"    {etape:<12} {v['p50']:>8.0f}ms {v['p95']:>8.0f}ms {v['max']:>8.0f}ms")
    lc = rapport["longueur_car"]
    if lc:
        print(f"    longueur      {lc['p50']:>8.0f}c  {lc['p95']:>8.0f}c  {lc['max']:>8.0f}c")
    a = rapport["abstention"]
    print(f"    abstentions   {a['attendue_et_faite']}/{a['attendue']} correctes · "
          f"{a['a_tort']} à tort")
    if rapport["n_echecs"]:
        print(f"    ÉCHECS        {rapport['n_echecs']}")
    if rapport["alertes_langue"]:
        print(f"    alertes langue {rapport['alertes_langue']}")

    r = rapport["retrieval"]
    print("\n  EXACTE — identifiants de passages comparés")
    print(f"    rappel moyen des passages de référence : "
          f"{r['rappel_moyen'] if r['rappel_moyen'] is not None else '—'}")
    print(f"    rang médian du premier passage utile   : {r['rang_median_utile'] or '—'}")
    print(f"    questions sans aucun passage utile     : {r['jamais_trouve']}")

    c, f = rapport["couverture"], rapport["fidelite"]
    print("\n  JUGÉE — un modèle décide, avec ses biais")
    print(f"    faits essentiels portés       : {c['essentiels']   if c['essentiels'] is not None else '—'}")
    print(f"    faits complémentaires portés  : {c['complementaires'] if c['complementaires'] is not None else '—'}"
          "    (doit chuter : c'est le prix de la brièveté)")
    print(f"    contrôle lexical parallèle    : {c['controle_lexical'] if c['controle_lexical'] is not None else '—'}"
          "    (un écart durable = juge qui dérive)")
    print(f"    contradictions / inventions   : {f['contradictions']} / {f['inventions']}")

    fautifs = [e for e in rapport["par_question"] if e.get("invente") or e.get("contredit")]
    if fautifs:
        print("\n    à regarder :")
        for e in fautifs:
            print(f"      q{e['id']} — {e.get('detail') or 'contradiction signalée'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("fichiers", nargs="*", help="fichiers bruts à noter")
    p.add_argument("--tous", action="store_true", help="noter tous les fichiers bruts")
    args = p.parse_args()

    if not REFERENCES.exists():
        sys.exit("benchmark/references.json absent.")
    refs = {r["id"]: r for r in json.loads(REFERENCES.read_text())}

    non_valides = sum(1 for r in refs.values() if not r.get("valide"))
    if non_valides:
        print(f"⚠️  {non_valides}/{len(refs)} références ne sont PAS validées à la main.\n"
              f"   Les mesures de qualité ci-dessous comparent le système à des\n"
              f"   réponses écrites par un modèle. À prendre comme un ordre de\n"
              f"   grandeur, pas comme une note.\n")

    from config import settings
    from groq import Groq
    if not settings.GROQ_API_KEY:
        sys.exit("GROQ_API_KEY absente — le juge ne peut pas tourner.")
    client = Groq(api_key=settings.GROQ_API_KEY)

    cibles = ([DOSSIER / f for f in args.fichiers] if args.fichiers
              else sorted(DOSSIER.glob("brut_*.json")) if args.tous else [])
    if not cibles:
        sys.exit("Rien à noter — passez un fichier ou --tous.")

    for chemin in cibles:
        if not chemin.exists():
            print(f"introuvable : {chemin.name}"); continue
        rapport = noter(chemin, refs, client, settings.GROQ_MODEL)
        afficher(rapport)
        sortie = chemin.with_name(chemin.name.replace("brut_", "rapport_"))
        sortie.write_text(json.dumps(rapport, ensure_ascii=False, indent=2))
        print(f"\n  → {sortie.relative_to(RACINE)}\n")


if __name__ == "__main__":
    main()
