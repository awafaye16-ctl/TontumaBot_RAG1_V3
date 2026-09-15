#!/usr/bin/env python3
"""Test de liaison Backend ↔ IA — rejoue la séquence complète du contrat.

Ce script fait ce que fera le backend Spring Boot, dans le même ordre, avec les
mêmes payloads :

    dépôt MinIO → notification from-storage → question → vérification des sources

Il sert à deux choses : valider la liaison avant que le backend ne soit branché,
et servir de test d'acceptation reproductible — s'il passe et que le backend
échoue, l'écart est côté backend, et réciproquement.

Exemples
--------
  # Chaîne complète, avec dépôt sur MinIO (nécessite un accès en écriture)
  python outils/tester_liaison.py \\
      --ia http://192.168.2.108:8008 \\
      --minio http://192.168.1.50:9000 --bucket tontuma-documents \\
      --access-key ADMIN --secret-key SECRET --deposer

  # L'objet a déjà été déposé par le backend : on ne fait que notifier
  python outils/tester_liaison.py --ia ... --minio ... --access-key ... \\
      --secret-key ... --bucket ... --objet "org/procedure.md"

  # Sans stockage objet : /ask et ingestion directe seulement
  python outils/tester_liaison.py --ia http://192.168.2.108:8008
"""
import argparse
import json
import sys
import time
import uuid
import urllib.error
import urllib.request

# Deux organisations : la seconde n'existe que pour prouver l'isolation.
ORG_A = "3f2a8c10-7b41-4e9d-9c22-1a5e6d0b8f31"
ORG_B = "9b7d1e44-2c08-4a17-b3f5-6e2d9c447a10"

DOC_A = (
    "Pour obtenir une carte d'identite nationale a la mairie de Thies, presentez "
    "un extrait de naissance, deux photos d'identite et le recu de paiement. "
    "Les frais s'elevent a 5000 francs CFA. Le delai de delivrance est de quinze "
    "jours ouvrables. Le guichet est ouvert du lundi au vendredi, de 8h a 15h."
)
DOC_A_V2 = (
    "Pour obtenir une carte d'identite nationale a la mairie de Thies, presentez "
    "un extrait de naissance, deux photos d'identite et le recu de paiement. "
    "Depuis la reforme, les frais passent a 7000 francs CFA et le delai a vingt "
    "jours ouvrables. Le guichet est ouvert du lundi au vendredi, de 8h a 15h."
)
DOC_B = (
    "Pour obtenir une copie de votre dossier medical, adressez une demande ecrite "
    "au service des archives de l'hopital Fann, muni de votre piece d'identite. "
    "La remise se fait sous huit jours ouvrables, du lundi au vendredi de 8h a 16h."
)

_echecs: list[str] = []
_t0 = time.time()


def titre(t: str) -> None:
    print(f"\n\033[1m── {t} \033[0m" + "─" * max(0, 60 - len(t)))


def ok(nom: str, condition: bool, detail: str = "") -> bool:
    marque = "\033[32m ✓\033[0m" if condition else "\033[31m ✗\033[0m"
    print(f"{marque} {nom}" + (f"\n     → {detail}" if detail and not condition else ""))
    if not condition:
        _echecs.append(nom)
    return condition


def info(texte: str) -> None:
    print(f"   \033[2m{texte}\033[0m")


# =============================================================================
#  HTTP
# =============================================================================

def appel(methode: str, url: str, corps=None, form=None, timeout=140):
    """Retourne (code, objet_json_ou_texte)."""
    donnees, entetes = None, {}
    if corps is not None:
        donnees = json.dumps(corps).encode()
        entetes["Content-Type"] = "application/json"
    elif form is not None:
        frontiere = "----tontuma" + uuid.uuid4().hex
        morceaux = []
        for cle, val in form.items():
            morceaux.append(f"--{frontiere}\r\nContent-Disposition: form-data; "
                            f'name="{cle}"\r\n\r\n{val}\r\n')
        morceaux.append(f"--{frontiere}--\r\n")
        donnees = "".join(morceaux).encode()
        entetes["Content-Type"] = f"multipart/form-data; boundary={frontiere}"

    req = urllib.request.Request(url, data=donnees, headers=entetes, method=methode)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            brut = r.read().decode()
            try:
                return r.status, json.loads(brut)
            except json.JSONDecodeError:
                return r.status, brut
    except urllib.error.HTTPError as e:
        brut = e.read().decode()
        try:
            return e.code, json.loads(brut)
        except json.JSONDecodeError:
            return e.code, brut
    except Exception as e:  # noqa: BLE001
        return 0, str(e)


def demander(ia: str, organisation: str, question: str, timeout=200) -> dict | None:
    """POST /ask et consomme le flux SSE, comme le fera WebClient.

    Le découpage se fait sur '\\n\\n' en conservant le fragment incomplet : un
    événement peut arriver coupé en deux lectures réseau.
    """
    corps = json.dumps({"question": question, "organization_id": organisation}).encode()
    req = urllib.request.Request(
        ia.rstrip("/") + "/ask", data=corps, method="POST",
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    resultat, etapes = None, []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        tampon = ""
        while True:
            paquet = r.read1(4096) if hasattr(r, "read1") else r.read(4096)
            if not paquet:
                break
            tampon += paquet.decode("utf-8", "replace")
            blocs = tampon.split("\n\n")
            tampon = blocs.pop()                     # fragment incomplet conservé
            for bloc in blocs:
                if not bloc.strip():
                    continue
                nom = bloc.split("event: ", 1)[1].split("\n", 1)[0]
                data = json.loads(bloc.split("data: ", 1)[1])
                if nom == "result":
                    resultat = data
                elif nom == "error":
                    print(f"     \033[31mevent: error → {data.get('message')}\033[0m")
                elif nom == "status":
                    etapes.append(data.get("step"))
    if etapes:
        info("étapes SSE : " + " → ".join(etapes))
    return resultat


# =============================================================================
#  Étapes du test
# =============================================================================

def etape_sante(ia: str) -> bool:
    titre("1. Service IA")
    code, d = appel("GET", ia.rstrip("/") + "/health", timeout=15)
    if not ok("/health répond", code == 200, f"code={code} · {d}"):
        return False
    ok("version 3.1.x", str(d.get("version", "")).startswith("3.1"), d.get("version"))
    ok("clé LLM configurée", d.get("llm_ready") is True,
       "llm_ready=false → le service répondra en mode extractif")
    info(f"backend de calcul : {d.get('device', {}).get('auto')}")
    info(f"organisations : {d.get('organizations')}")
    stock = d.get("storage", {})
    info(f"stockage objet : {stock}")
    if stock.get("configure") and stock.get("joignable") is False:
        ok("stockage objet joignable depuis l'IA", False,
           f"l'IA ne joint pas MinIO : {stock.get('erreur')}")
    return True


def etape_minio(args) -> str | None:
    """Dépose le document de test sur MinIO. Retourne la clé de l'objet."""
    titre("2. Stockage objet (MinIO)")
    if not args.minio:
        info("non configuré — étape ignorée (--minio absent)")
        return None

    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        ok("boto3 disponible", False, "pip install boto3")
        return None

    client = boto3.client(
        "s3", endpoint_url=args.minio,
        aws_access_key_id=args.access_key, aws_secret_access_key=args.secret_key,
        region_name=args.region,
        # Path-style : MinIO ne sert pas l'adressage par sous-domaine.
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"},
                      connect_timeout=5, read_timeout=30,
                      retries={"max_attempts": 2}),
    )
    try:
        client.list_buckets()
        ok(f"MinIO joignable depuis cette machine ({args.minio})", True)
    except Exception as e:  # noqa: BLE001
        ok(f"MinIO joignable depuis cette machine ({args.minio})", False, str(e)[:180])
        info("si l'endpoint contient 'minio' ou 'localhost', c'est un nom interne au")
        info("Docker de l'autre machine : il faut l'IP de l'hôte, pas le nom du service.")
        return None

    cle = args.objet or f"{ORG_A}/test-liaison-{uuid.uuid4().hex[:8]}.md"
    if args.deposer:
        try:
            client.put_object(Bucket=args.bucket, Key=cle,
                              Body=DOC_A.encode(), ContentType="text/markdown")
            ok(f"dépôt de l'objet de test ({cle})", True)
        except Exception as e:  # noqa: BLE001
            ok(f"dépôt de l'objet de test ({cle})", False, str(e)[:180])
            info("accès en écriture requis pour --deposer ; sinon, faites déposer")
            info("l'objet par le backend et passez sa clé avec --objet.")
            return None
    else:
        try:
            client.head_object(Bucket=args.bucket, Key=cle)
            ok(f"objet présent sur MinIO ({cle})", True)
        except Exception as e:  # noqa: BLE001
            ok(f"objet présent sur MinIO ({cle})", False, str(e)[:180])
            return None
    return cle


def etape_ingestion(ia: str, args, cle: str | None) -> str | None:
    titre("3. Ingestion")
    base = ia.rstrip("/") + f"/admin/organizations/{ORG_A}/documents"
    doc_id = str(uuid.uuid4())

    if cle:
        code, d = appel("POST", base + "/from-storage", corps={
            "documentId": doc_id, "bucket": args.bucket, "objectKey": cle,
            "title": "Carte d'identité — Thiès", "category": "procedure"})
        if not ok("notification from-storage acceptée", code == 200,
                  f"code={code} · {d}"):
            if code == 500:
                info("500 → stockage non configuré CÔTÉ IA : renseignez S3_* dans .env")
            if code == 502:
                info("502 → l'IA ne joint pas MinIO. Elle l'atteint depuis SA machine,")
                info("      pas depuis la vôtre : vérifiez S3_ENDPOINT dans son .env.")
            return None
        ok("document_id restitué à l'identique", d.get("document_id") == doc_id,
           f"{d.get('document_id')} ≠ {doc_id}")
        ok("premier dépôt → replaced=false", d.get("replaced") is False)
        info(f"{d.get('chunks')} fragment(s) indexé(s)")
    else:
        code, d = appel("POST", base, form={
            "text": DOC_A, "title": "Carte d'identité — Thiès",
            "document_id": doc_id, "category": "procedure"})
        if not ok("ingestion directe acceptée", code == 200, f"code={code} · {d}"):
            return None
        info(f"{d.get('chunks')} fragment(s) indexé(s) (voie multipart)")

    # Organisation B, pour le test d'isolation
    code, _ = appel("POST", ia.rstrip("/") + f"/admin/organizations/{ORG_B}/documents",
                    form={"text": DOC_B, "title": "Dossier médical — Fann",
                          "document_id": "doc-hopital-fann", "category": "procedure"})
    ok("seconde organisation alimentée (pour l'isolation)", code == 200)

    code, d = appel("GET", base)
    ok("listage de l'organisation", code == 200 and d.get("total_documents", 0) >= 1,
       str(d))
    return doc_id


def etape_republication(ia: str, doc_id: str) -> None:
    titre("4. Republication (le point qui piégeait le contrat)")
    base = ia.rstrip("/") + f"/admin/organizations/{ORG_A}/documents"
    code, d = appel("POST", base, form={
        "text": DOC_A_V2, "title": "Carte d'identité — Thiès",
        "document_id": doc_id, "category": "procedure"})
    ok("republication acceptée", code == 200, f"code={code} · {d}")
    ok("replaced=true", d.get("replaced") is True if code == 200 else False)

    code, d = appel("GET", base)
    ok("un seul document — aucune duplication",
       code == 200 and d.get("total_documents") == 1,
       f"total_documents={d.get('total_documents') if code == 200 else '?'}")


def etape_questions(ia: str, doc_id: str) -> None:
    titre("5. Questions et traçabilité")

    r = demander(ia, ORG_A, "Quels sont les frais de la carte d'identite ?")
    if not ok("réponse reçue sur /ask", r is not None):
        return
    print(f"     réponse : {r['response'][:110]}")
    ok("champ sources présent", isinstance(r.get("sources"), list))
    ok("sources non vides", len(r.get("sources") or []) > 0)
    if r.get("sources"):
        s = r["sources"][0]
        print(f"     sources : {json.dumps(r['sources'], ensure_ascii=False)}")
        ok("document_id de la source = celui fourni à l'ingestion",
           s.get("document_id") == doc_id, f"{s.get('document_id')} ≠ {doc_id}")
        ok("sources bien formées",
           {"document_id", "title", "category", "chunk_id", "rank", "score"} <= set(s))
    chiffres = "".join(c for c in r["response"] if c.isdigit())
    ok("c'est la version republiée qui répond (7000, pas 5000)",
       "7000" in chiffres and "5000" not in chiffres, f"chiffres vus : {chiffres}")
    ok("trace présente (à journaliser, jamais à relayer au navigateur)",
       isinstance(r.get("trace"), dict))
    ok("confidence absent (le backend doit l'accepter optionnel)",
       "confidence" not in r)

    titre("6. Isolation entre organisations")
    r = demander(ia, ORG_A, "Comment obtenir une copie de mon dossier medical a l'hopital Fann ?")
    if ok("réponse reçue", r is not None):
        ids = [s.get("document_id") for s in (r.get("sources") or [])]
        print(f"     réponse : {r['response'][:110]}")
        ok("A ne cite aucun document de B", "doc-hopital-fann" not in ids, str(ids))
        ok("« Fann » absent de la réponse de A", "fann" not in r["response"].lower(),
           r["response"][:90])

    r = demander(ia, ORG_B, "Quels sont les frais de la carte d'identite a Thies ?")
    if ok("réponse reçue pour B", r is not None):
        ids = [s.get("document_id") for s in (r.get("sources") or [])]
        ok("B ne cite aucun document de A", doc_id not in ids, str(ids))

    vide = str(uuid.uuid4())
    r = demander(ia, vide, "Quels documents pour une carte d'identite ?")
    if ok("réponse reçue pour une organisation vide", r is not None):
        ok("organisation vide → sources vides", r.get("sources") == [],
           str(r.get("sources")))
        ok("aucun repli sur le jeu de démonstration",
           r.get("trace", {}).get("retrieval", {}).get("source") == "aucun document",
           str(r.get("trace", {}).get("retrieval", {}).get("source")))


def etape_erreurs(ia: str, args) -> None:
    titre("7. Codes d'erreur du contrat")
    for valeur, attendu, libelle in [
        (None, 400, "organization_id absent → 400"),
        ("pas-un-uuid", 400, "organization_id mal formé → 400"),
        ("../../etc/passwd", 400, "tentative de traversée de chemin → 400"),
    ]:
        code, d = appel("POST", ia.rstrip("/") + "/ask",
                        corps={"question": "test", "organization_id": valeur}, timeout=20)
        ok(libelle, code == attendu, f"code={code} · {d}")

    code, _ = appel("DELETE", ia.rstrip("/") +
                    f"/admin/organizations/{ORG_A}/documents/inexistant", timeout=20)
    ok("document inconnu à la suppression → 404", code == 404, f"code={code}")

    if args.minio:
        code, d = appel("POST", ia.rstrip("/") +
                        f"/admin/organizations/{ORG_A}/documents/from-storage",
                        corps={"documentId": str(uuid.uuid4()), "bucket": args.bucket,
                               "objectKey": "objet/qui/nexiste/pas.pdf"}, timeout=40)
        ok("objet MinIO introuvable → 404", code == 404, f"code={code} · {d}")


def etape_menage(ia: str, garder: bool) -> None:
    titre("8. Ménage")
    if garder:
        info("--garder : les organisations de test sont conservées")
        return
    for org in (ORG_A, ORG_B):
        code, _ = appel("DELETE", ia.rstrip("/") + f"/admin/organizations/{org}", timeout=30)
        ok(f"organisation {org[:8]}… supprimée", code in (200, 404), f"code={code}")


# =============================================================================
#  Point d'entrée
# =============================================================================

def main() -> int:
    p = argparse.ArgumentParser(
        description="Test de liaison Backend ↔ IA (rejoue la séquence du contrat)",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--ia", default="http://localhost:8008",
                   help="URL du service IA (défaut : http://localhost:8008)")
    p.add_argument("--minio", help="endpoint MinIO — l'IP de l'hôte, pas un nom Docker")
    p.add_argument("--bucket", default="tontuma-documents")
    p.add_argument("--access-key", default="")
    p.add_argument("--secret-key", default="")
    p.add_argument("--region", default="us-east-1")
    p.add_argument("--objet", help="clé d'un objet déjà déposé par le backend")
    p.add_argument("--deposer", action="store_true",
                   help="déposer le document de test (nécessite un accès en écriture)")
    p.add_argument("--garder", action="store_true",
                   help="ne pas supprimer les organisations de test à la fin")
    args = p.parse_args()

    print(f"\033[1mTest de liaison Backend ↔ IA\033[0m")
    print(f"  IA    : {args.ia}")
    print(f"  MinIO : {args.minio or '(non configuré — voie from-storage ignorée)'}")

    if not etape_sante(args.ia):
        print("\n\033[31mService IA injoignable — rien d'autre ne peut être testé.\033[0m")
        return 1

    cle = etape_minio(args)
    doc_id = etape_ingestion(args.ia, args, cle)
    if doc_id:
        etape_republication(args.ia, doc_id)
        etape_questions(args.ia, doc_id)
    etape_erreurs(args.ia, args)
    etape_menage(args.ia, args.garder)

    print("\n" + "=" * 64)
    duree = time.time() - _t0
    if _echecs:
        print(f"\033[31m{len(_echecs)} ÉCHEC(S)\033[0m en {duree:.0f} s :")
        for e in _echecs:
            print(f"   • {e}")
        return 1
    print(f"\033[32mTout est vert\033[0m — liaison conforme au contrat ({duree:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
