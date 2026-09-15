"""Produit des réponses de référence candidates, à valider par un humain.

Pourquoi ce script existe : sans vérité de référence, on ne peut mesurer que la
latence. Et une référence dérivée de ce que le pipeline a répondu ne mesure
rien du tout — elle évaluerait le système contre lui-même.

La référence est donc écrite à partir du corpus ENTIER (32 chunks, ~3 700
jetons : il tient dans une fenêtre de contexte), pas des passages que la
recherche fait remonter. Le retrieval devient alors mesurable : on peut
demander si la recherche a bien ramené les chunks sur lesquels la référence
s'appuie.

Le fichier produit porte `valide: false` sur chaque entrée. Tant que ce drapeau
n'est pas levé à la main, ce n'est PAS une vérité de référence : c'est la
proposition d'un modèle, avec les erreurs d'un modèle.

Usage :
    venv/bin/python benchmark/generer_references.py
    → benchmark/references.json
"""
import json
import os
import sys
import time
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "src"))
os.chdir(RACINE)

from config import settings          # noqa: E402
import vectorstore                   # noqa: E402
from translation.nllb import wolof_to_french  # noqa: E402

SORTIE = RACINE / "benchmark" / "references.json"

# Plafond propre à ce script, sans rapport avec celui du service. En mode JSON,
# le modèle doit produire un document complet et valide : si sa trace de
# raisonnement épuise le budget avant, l'appel échoue en bloc — c'est ce qui
# était arrivé à q1 avec les 800 jetons du service. Ici, aucun usager n'attend :
# on paie large.
MAX_JETONS = int(os.getenv("REFERENCE_MAX_TOKENS", "4000"))

CONSIGNE = """Tu construis un jeu d'évaluation pour un assistant hospitalier sénégalais.

On te donne le corpus documentaire COMPLET, découpé en passages numérotés, puis
une question d'usager. Tu ne réponds pas à l'usager : tu écris la réponse de
référence qui servira à noter le système.

Rends un objet JSON avec exactement ces clés :

  "repondable"  : true si le corpus contient de quoi répondre, false sinon.
  "reponse"     : la réponse idéale en français, factuelle, 1 à 4 phrases,
                  uniquement à partir des passages. Chaîne vide si repondable
                  est false.
  "passages"    : la liste des numéros de passages qui soutiennent la réponse,
                  du plus au moins important. Liste vide si repondable est false.
  "motif"       : si repondable est false, ce qui manque au corpus. Sinon "".

Règles :
- N'invente aucun chiffre, lieu, horaire ou montant absent des passages.
- Ne cite que des passages réellement utilisés.
- Une question hors du domaine du corpus doit avoir repondable = false :
  l'abstention est la bonne réponse, et le jeu doit pouvoir le vérifier.
"""


def corpus_numerote() -> tuple[str, list[str]]:
    """Le corpus entier en texte numéroté, et la table numéro → identifiant."""
    c = vectorstore._get_corpus(settings.ORGANISATION_OUTILS)
    lignes, ids = [], []
    for i, (cid, texte, meta) in enumerate(zip(c["ids"], c["texts"], c["metas"]), 1):
        ids.append(cid)
        lignes.append(f"[Passage {i}] (source : {meta.get('source', '?')})\n{texte}")
    return "\n\n".join(lignes), ids


def interroger(client, corpus: str, question_wo: str, question_fr: str) -> dict:
    reponse = client.chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=[
            {"role": "system", "content": CONSIGNE},
            {"role": "user", "content":
                f"CORPUS COMPLET :\n{corpus}\n\n"
                f"Question de l'usager (wolof) : {question_wo}\n"
                f"Traduction automatique (français) : {question_fr}"},
        ],
        temperature=0,
        max_tokens=MAX_JETONS,
        response_format={"type": "json_object"},
    )
    return json.loads(reponse.choices[0].message.content or "{}")


def existantes() -> dict:
    """Références déjà produites, indexées par id de question."""
    if not SORTIE.exists():
        return {}
    try:
        return {e["id"]: e for e in json.loads(SORTIE.read_text())}
    except Exception:
        return {}


def main() -> None:
    if not settings.GROQ_API_KEY:
        sys.exit("GROQ_API_KEY absente de .env — impossible de générer les références.")

    # Par défaut on complète : un appel peut échouer isolément, et refaire les
    # 19 autres coûterait treize minutes pour rien. `--tout` régénère l'ensemble,
    # ce qui vaut mieux quand les réglages de génération ont changé : un jeu
    # d'évaluation doit être homogène.
    tout   = "--tout" in sys.argv
    deja   = {} if tout else existantes()
    # Une référence relue et validée à la main ne doit jamais être écrasée.
    gardees = {i: e for i, e in deja.items() if e.get("valide") or not tout}
    if deja:
        print(f"[reprise] {len(deja)} référence(s) déjà présentes"
              + (" — ignorées (--tout)" if tout else ""), flush=True)

    from groq import Groq
    client = Groq(api_key=settings.GROQ_API_KEY)

    corpus, ids = corpus_numerote()
    print(f"[corpus] {len(ids)} passages, {len(corpus)} caractères", flush=True)

    questions = json.loads((RACINE / "questions_test_wo.json").read_text())
    sorties = []

    for q in questions:
        if q["id"] in gardees:
            e = gardees[q["id"]]
            sorties.append(e)
            marque = " (validée à la main)" if e.get("valide") else ""
            print(f"  q{q['id']:>2} conservée{marque}", flush=True)
            continue
        t0 = time.perf_counter()
        question_wo = q["question"]
        question_fr, _ = wolof_to_french(question_wo)
        try:
            r = interroger(client, corpus, question_wo, question_fr)
        except Exception as e:
            print(f"  q{q['id']:>2} ÉCHEC {type(e).__name__}: {e}", flush=True)
            continue

        numeros = [n for n in r.get("passages", []) if isinstance(n, int) and 1 <= n <= len(ids)]
        sorties.append({
            "id":            q["id"],
            "question_wo":   question_wo,
            # Traduction du pipeline, montrée pour que la relecture humaine
            # repère une question déformée avant qu'elle ne fausse la référence.
            "question_fr":   question_fr,
            "repondable":    bool(r.get("repondable")),
            "reponse":       (r.get("reponse") or "").strip(),
            "motif":         (r.get("motif") or "").strip(),
            "chunks_ids":    [ids[n - 1] for n in numeros],
            "chunks_numeros": numeros,
            # Tant que ce drapeau est false, ce n'est pas une vérité de
            # référence mais la proposition d'un modèle.
            "valide":        False,
        })
        etat = "répondable" if sorties[-1]["repondable"] else "ABSTENTION attendue"
        print(f"  q{q['id']:>2} {etat:20} {len(numeros)} passage(s)  "
              f"({time.perf_counter() - t0:.1f}s)", flush=True)

    sorties.sort(key=lambda e: e["id"])
    SORTIE.write_text(json.dumps(sorties, ensure_ascii=False, indent=2))
    n_rep = sum(1 for s in sorties if s["repondable"])
    print(f"\n→ {SORTIE.relative_to(RACINE)}")
    print(f"   {len(sorties)} questions · {n_rep} répondables · "
          f"{len(sorties) - n_rep} abstentions attendues")
    print("   Toutes marquées valide=false : à relire avant de servir de référence.")


if __name__ == "__main__":
    main()
