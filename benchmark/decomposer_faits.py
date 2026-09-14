"""Décompose chaque réponse de référence en faits atomiques étiquetés.

Pourquoi : le système a pour consigne de répondre en trois phrases, quand la
référence décrit une réponse complète. Comparer les deux par recouvrement de
mots punirait le système d'avoir obéi.

On sépare donc ce que ce recouvrement confond :

  essentiel      sans ce fait, l'usager ne peut pas agir. Une réponse brève
                 DOIT le porter — si la couverture des essentiels décroche,
                 c'est que la consigne de brièveté est trop serrée.
  complementaire utile mais second. Sa couverture va chuter avec la brièveté :
                 c'est le prix consenti, pas un défaut. On le mesure pour
                 savoir ce qu'on paie.

Le fichier est enrichi sur place : une seule relecture humaine pour l'ensemble.
Les entrées déjà décomposées sont conservées, celles marquées `valide` ne sont
jamais retouchées.

Usage :
    venv/bin/python benchmark/decomposer_faits.py [--tout]
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

FICHIER    = RACINE / "benchmark" / "references.json"
MAX_JETONS = int(os.getenv("REFERENCE_MAX_TOKENS", "4000"))

CONSIGNE = """Tu prépares la grille de notation d'un assistant hospitalier.

On te donne une question d'usager et la réponse de référence complète. Découpe
cette réponse en faits atomiques et classe-les.

Rends un objet JSON avec une seule clé "faits", liste d'objets :

  "fait"       : l'information, en une phrase courte et autonome, sans pronom
                 renvoyant à un autre fait.
  "rang"       : 1 pour le plus important, puis 2, 3... sans ex aequo.
  "importance" : "essentiel" ou "complementaire".
  "indices"    : 2 à 5 mots ou nombres qui doivent apparaître, sous une forme
                 ou une autre, dans toute réponse portant ce fait.

Règle de classement, à appliquer strictement :

  AU PLUS TROIS faits "essentiel". Ce sont ceux sans lesquels l'usager se
  déplace pour rien ou repart bredouille : où aller, quoi apporter, combien
  payer, quel délai, quelle condition décisive.

  Tout le reste est "complementaire". En particulier, est TOUJOURS
  complementaire ce que le personnel fait une fois l'usager sur place —
  « l'agent enregistre le dossier », « le médecin examine le patient »,
  « le patient attend son tour ». L'usager n'a rien à en faire : il le
  découvrira en arrivant. Ce sont des narrations, pas des instructions.

Regroupe ce qui s'accomplit d'un seul geste : les pièces à apporter forment UN
fait qui les énumère, pas un fait par pièce.

Une réponse de trois phrases doit pouvoir porter tous les faits essentiels. Si
tu en comptes plus de trois, c'est que tu as classé une narration comme une
instruction : reprends.
"""



def interroger(client, question: str, reference: str) -> list[dict]:
    r = client.chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=[
            {"role": "system", "content": CONSIGNE},
            {"role": "user", "content": f"Question : {question}\n\nRéponse de référence :\n{reference}"},
        ],
        temperature=0,
        max_tokens=MAX_JETONS,
        response_format={"type": "json_object"},
    )
    faits = json.loads(r.choices[0].message.content or "{}").get("faits", [])
    propres = []
    for f in faits:
        if not isinstance(f, dict) or not (f.get("fait") or "").strip():
            continue
        propres.append({
            "fait":       f["fait"].strip(),
            "rang":       int(f["rang"]) if str(f.get("rang", "")).isdigit() else 99,
            "importance": "essentiel" if f.get("importance") == "essentiel" else "complementaire",
            "indices":    [str(i).strip() for i in (f.get("indices") or []) if str(i).strip()],
        })
    propres.sort(key=lambda f: f["rang"])

    # Filet : le modèle déborde parfois son propre plafond. Le rang fait alors
    # autorité — au-delà du troisième, c'est complémentaire, quoi qu'il en dise.
    for i, f in enumerate(propres):
        if i >= 3 and f["importance"] == "essentiel":
            f["importance"] = "complementaire"
            f["reclasse"] = True
    return propres


def main() -> None:
    if not settings.GROQ_API_KEY:
        sys.exit("GROQ_API_KEY absente de .env.")
    if not FICHIER.exists():
        sys.exit("benchmark/references.json absent — lancez d'abord generer_references.py.")

    from groq import Groq
    client = Groq(api_key=settings.GROQ_API_KEY)

    tout = "--tout" in sys.argv
    refs = json.loads(FICHIER.read_text())

    for r in refs:
        if not r["repondable"]:
            r["faits"] = []            # rien à couvrir : l'abstention est la réponse
            continue
        if r.get("valide"):
            print(f"  q{r['id']:>2} validée à la main — inchangée", flush=True)
            continue
        if r.get("faits") and not tout:
            print(f"  q{r['id']:>2} déjà décomposée ({len(r['faits'])} faits)", flush=True)
            continue

        t0 = time.perf_counter()
        try:
            faits = interroger(client, r["question_fr"], r["reponse"])
        except Exception as e:
            print(f"  q{r['id']:>2} ÉCHEC {type(e).__name__}: {str(e)[:90]}", flush=True)
            continue
        r["faits"] = faits
        ess = sum(1 for f in faits if f["importance"] == "essentiel")
        print(f"  q{r['id']:>2} {len(faits)} faits — {ess} essentiels, "
              f"{len(faits) - ess} complémentaires  ({time.perf_counter() - t0:.1f}s)", flush=True)

    FICHIER.write_text(json.dumps(refs, ensure_ascii=False, indent=2))

    avec   = [r for r in refs if r.get("faits")]
    ess    = sum(1 for r in avec for f in r["faits"] if f["importance"] == "essentiel")
    total  = sum(len(r["faits"]) for r in avec)
    print(f"\n→ {FICHIER.relative_to(RACINE)}")
    print(f"   {len(avec)} réponses décomposées · {total} faits dont {ess} essentiels")
    trop = [r["id"] for r in avec
            if sum(1 for f in r["faits"] if f["importance"] == "essentiel") > 4]
    if trop:
        print(f"   ⚠️  plus de 4 essentiels sur q{', q'.join(map(str, trop))} — "
              f"à resserrer à la relecture, ou la brièveté sera jugée trop sévèrement")


if __name__ == "__main__":
    main()
