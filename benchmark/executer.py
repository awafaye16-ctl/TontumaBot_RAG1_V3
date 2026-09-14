"""Exécute le jeu de questions et enregistre les résultats bruts.

Ce script ne juge rien : il collecte. La notation vit dans `noter.py`, et cette
séparation est délibérée — un passage coûte des minutes de calcul et des appels
payants, alors qu'une grille de notation se corrige souvent. On veut pouvoir
renoter d'anciens résultats sans les reproduire.

Chaque fichier produit embarque la configuration qui l'a produit : sans elle, un
chiffre de latence ou de longueur ne veut rien dire six semaines plus tard.

Usage :
    venv/bin/python benchmark/executer.py                  # 1 passage, sans TTS
    venv/bin/python benchmark/executer.py --repetitions 3  # pour des percentiles
    venv/bin/python benchmark/executer.py --tts            # avec synthèse vocale
    venv/bin/python benchmark/executer.py --phrases 2,3,4  # balayage de la brièveté
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "src"))
os.chdir(RACINE)

DOSSIER   = RACINE / "benchmark" / "resultats"
QUESTIONS = RACINE / "benchmark" / "references.json"


def configuration() -> dict:
    """Tout ce qui, en changeant, invaliderait la comparaison avec un autre passage."""
    from config import settings
    import device

    def git(*args):
        try:
            return subprocess.check_output(["git", *args], text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    return {
        "horodatage":       datetime.now().isoformat(timespec="seconds"),
        "commit":           git("rev-parse", "--short", "HEAD"),
        "arbre_propre":     git("status", "--porcelain") == "",
        "machine":          f"{platform.system()} {platform.machine()}",
        "backends":         device.infos(),
        "llm_provider":     settings.LLM_PROVIDER,
        "llm_modele":       settings.GROQ_MODEL,
        "llm_max_phrases":  settings.LLM_MAX_PHRASES,
        "llm_max_tokens":   settings.LLM_MAX_TOKENS,
        "reranker_top_k":   settings.RERANKER_TOP_K,
        "tts_n_steps":      settings.TTS_N_STEPS,
        "tts_chunk_chars":  settings.TTS_CHUNK_CHARS,
        "n_chunks_indexes": None,      # rempli après chargement du vectorstore
    }


def duree_audio(chemin: str | None) -> float | None:
    if not chemin or not Path(chemin).exists():
        return None
    try:
        import soundfile as sf
        return round(sf.info(chemin).duration, 2)
    except Exception:
        return None


def un_passage(questions: list[dict], avec_tts: bool, repetitions: int,
               prechauffer: bool = True) -> list[dict]:
    from pipeline import answer as pipeline_answer
    import vectorstore

    # Le premier appel charge les modèles et compile les noyaux de calcul :
    # mesuré, il porte la traduction à 17,8 s contre 2 s ensuite. L'inclure
    # dans les mesures ferait passer un coût de démarrage pour une latence de
    # service. On le paie une fois, sans l'enregistrer.
    if prechauffer:
        t0 = time.perf_counter()
        try:
            pipeline_answer(questions[0]["question_wo"], provider=os.getenv("BENCH_PROVIDER", "groq"),
                            tts=False, lang_hint="wo")
            print(f"  [préchauffage] {time.perf_counter() - t0:.1f}s — non mesuré\n", flush=True)
        except Exception as e:
            print(f"  [préchauffage] échoué ({type(e).__name__}) — les premières "
                  f"mesures seront hautes\n", flush=True)

    resultats = []
    for rep in range(1, repetitions + 1):
        for q in questions:
            etapes = []
            t0 = time.perf_counter()
            sortie = RACINE / "benchmark" / "resultats" / f"audio_q{q['id']}_r{rep}.wav"
            try:
                r = pipeline_answer(
                    q["question_wo"],
                    provider  = os.getenv("BENCH_PROVIDER", "groq"),
                    tts       = avec_tts,
                    tts_out   = str(sortie) if avec_tts else None,
                    # La borne déclare toujours la langue : on reproduit le vrai
                    # chemin, indice compris, plutôt qu'un cas de laboratoire.
                    lang_hint = "wo",
                    progress  = lambda s, i: etapes.append({"etape": s, **i}),
                )
            except Exception as e:
                message = str(e)
                print(f"  r{rep} q{q['id']:>2} ÉCHEC {type(e).__name__}: {message[:80]}", flush=True)
                resultats.append({"id": q["id"], "repetition": rep,
                                  "echec": f"{type(e).__name__}: {e}"})
                # Un quota épuisé ne se répare pas en insistant : continuer
                # produirait cinquante échecs de plus et un fichier de résultats
                # trompeur, qui ressemble à une mesure sans en être une.
                if "rate limit" in message.lower() or "429" in message:
                    print("\n  ⛔ Quota du fournisseur atteint — passage interrompu.\n"
                          "     Les résultats partiels sont enregistrés, mais ils ne\n"
                          "     couvrent qu'une partie du jeu : ne pas les comparer à\n"
                          "     un passage complet.", flush=True)
                    return resultats
                continue
            duree = round(time.perf_counter() - t0, 2)
            trace = r.get("trace", {})

            resultats.append({
                "id":            q["id"],
                "repetition":    rep,
                "question_wo":   q["question_wo"],
                "question_fr":   trace.get("wolof_to_french", {}).get("result"),
                # La notation porte sur le français : c'est la langue pivot du
                # pipeline, et la référence y est écrite. Juger le wolof
                # mesurerait la traduction de sortie en même temps que le RAG.
                "response_fr":   r.get("response_fr", ""),
                "response_wo":   r.get("response_wo"),
                "chunks":        [c["id"] for c in trace.get("retrieval", {}).get("chunks", [])],
                "scores":        trace.get("retrieval", {}).get("reranker_scores", []),
                "intent":        trace.get("intent"),
                "alerte_langue": trace.get("alerte_langue"),
                "nombres":       trace.get("french_to_wolof", {}).get("nombres"),
                "duree_s":       duree,
                "latences_ms": {
                    "wo_fr":     trace.get("wolof_to_french", {}).get("latency_ms"),
                    "intent":    trace.get("intent_latency_ms"),
                    "retrieval": trace.get("retrieval", {}).get("latency_total_ms"),
                    "rerank":    trace.get("retrieval", {}).get("latency_rerank_ms"),
                    "llm":       trace.get("llm", {}).get("latency_ms"),
                    "fr_wo":     trace.get("french_to_wolof", {}).get("latency_ms"),
                    "tts":       trace.get("tts", {}).get("latency_ms"),
                    "total":     trace.get("total_latency_ms"),
                },
                "tts": {
                    "morceaux":   trace.get("tts", {}).get("n_morceaux"),
                    "duree_audio": duree_audio(str(sortie) if avec_tts else None),
                } if avec_tts else None,
                "etapes": [e["etape"] for e in etapes],
            })
            marque = "⚠" if resultats[-1]["alerte_langue"] else " "
            print(f"  r{rep} q{q['id']:>2} {marque} {duree:6.1f}s  "
                  f"{len(resultats[-1]['response_fr']):>4} car.  "
                  f"{len(resultats[-1]['chunks'])} chunks", flush=True)
    return resultats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repetitions", type=int, default=1,
                   help="passages successifs — au-delà de 1, les percentiles ont un sens")
    p.add_argument("--tts", action="store_true", help="produire aussi l'audio")
    p.add_argument("--phrases", default="",
                   help="balayage de LLM_MAX_PHRASES, ex. 2,3,4 — un fichier par valeur")
    p.add_argument("--questions", default="",
                   help="restreindre à certains ids, ex. 1,5,8 (mise au point)")
    p.add_argument("--sans-prechauffage", action="store_true",
                   help="mesurer aussi le premier appel, chargement des modèles compris")
    args = p.parse_args()

    if not QUESTIONS.exists():
        sys.exit("benchmark/references.json absent — lancez d'abord generer_references.py.")

    # Le balayage relance le script par sous-processus : les réglages sont lus à
    # l'import de config, un rechargement à chaud laisserait des modules tièdes.
    if args.phrases:
        for valeur in [v.strip() for v in args.phrases.split(",") if v.strip()]:
            print(f"\n═══ LLM_MAX_PHRASES={valeur} ═══", flush=True)
            env = {**os.environ, "LLM_MAX_PHRASES": valeur}
            cmd = [sys.executable, __file__, "--repetitions", str(args.repetitions)]
            if args.tts:
                cmd.append("--tts")
            subprocess.run(cmd, env=env, check=False)
        return

    questions = json.loads(QUESTIONS.read_text())
    if args.questions:
        voulus = {int(v) for v in args.questions.split(",") if v.strip().isdigit()}
        questions = [q for q in questions if q["id"] in voulus]
        if not questions:
            sys.exit(f"Aucune question parmi {sorted(voulus)}.")
    cfg = configuration()
    import vectorstore
    cfg["n_chunks_indexes"] = vectorstore.count()

    print(f"[config] commit {cfg['commit']} "
          f"{'(arbre propre)' if cfg['arbre_propre'] else '(MODIFICATIONS NON COMMITÉES)'} · "
          f"{cfg['llm_max_phrases']} phrases · {cfg['n_chunks_indexes']} chunks · "
          f"backend {cfg['backends']['auto']}", flush=True)
    print(f"[jeu] {len(questions)} questions × {args.repetitions} passage(s)"
          f"{' · TTS activé' if args.tts else ''}\n", flush=True)

    debut = time.perf_counter()
    resultats = un_passage(questions, args.tts, args.repetitions,
                           prechauffer=not args.sans_prechauffage)

    # Le bilan d'intégrité se calcule AVANT l'écriture : l'inverse a déjà coûté
    # un passage entier, exécuté puis perdu sur un UnboundLocalError au moment
    # d'enregistrer.
    echecs   = sum(1 for r in resultats if r.get("echec"))
    attendu  = len(questions) * args.repetitions
    complet  = len(resultats) == attendu and echecs == 0

    DOSSIER.mkdir(parents=True, exist_ok=True)
    nom = (f"brut_{datetime.now():%Y%m%d_%H%M}"
           f"_p{cfg['llm_max_phrases']}{'_tts' if args.tts else ''}.json")
    chemin = DOSSIER / nom
    chemin.write_text(json.dumps(
        {"configuration": cfg, "avec_tts": args.tts,
         "repetitions": args.repetitions,
         "prechauffage": not args.sans_prechauffage,
         # Sans ce drapeau, un passage tronqué par un quota se compare
         # innocemment à un passage entier.
         "complet": complet,
         "attendu": attendu,
         "resultats": resultats},
        ensure_ascii=False, indent=2))

    print(f"\n→ {chemin.relative_to(RACINE)}")
    print(f"   {len(resultats)}/{attendu} exécutions · {echecs} échec(s) · "
          f"{time.perf_counter() - debut:.0f}s au total"
          + ("" if complet else "  ⚠️ PASSAGE INCOMPLET"))
    print(f"   Notation : venv/bin/python benchmark/noter.py {chemin.name}")


if __name__ == "__main__":
    main()
