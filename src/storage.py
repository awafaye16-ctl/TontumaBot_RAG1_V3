"""Stockage objet (MinIO / S3) — lecture seule.

Le backend dépose le fichier sur MinIO puis notifie l'IA avec un pointeur
(`bucket` + `objectKey`) ; l'IA lit l'objet et l'indexe. Ce module ne fait que
la lecture : l'IA n'écrit JAMAIS sur le stockage objet, et le compte de service
qu'on lui donne doit être restreint en conséquence.

Pourquoi un module à part
-------------------------
L'ingestion ne doit pas savoir d'où vient un fichier. Elle reçoit un chemin
local ; que ce chemin vienne d'un téléversement direct ou d'un téléchargement
MinIO ne change rien à son travail. Les deux voies d'ingestion du contrat —
pointeur et envoi direct — partagent donc tout leur code en aval d'ici.
"""
import os
import sys
import tempfile
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import settings  # noqa: E402

from journal import journal

_log = journal("storage")

_client = None


# =============================================================================
#  Erreurs
# =============================================================================

class StockageIndisponible(RuntimeError):
    """MinIO injoignable ou identifiants refusés → 502. Rejouer avec backoff."""


class StockageNonConfigure(RuntimeError):
    """Ni endpoint ni identifiants renseignés côté IA → 500.

    Distinct de `StockageIndisponible` à dessein : là, rejouer ne servira à
    rien. C'est une erreur de déploiement, et le backend doit alerter plutôt
    que réessayer en boucle.
    """


class ObjetIntrouvable(FileNotFoundError):
    """L'objet n'existe pas dans ce bucket → 404.

    Le cas courant n'est pas une faute de frappe : c'est un POST de notification
    parti avant que le PUT sur MinIO ne soit terminé.
    """


class ObjetTropVolumineux(ValueError):
    """L'objet dépasse S3_MAX_MB → 400."""


# =============================================================================
#  Client
# =============================================================================

def get_client():
    """Client S3 partagé, construit à la première utilisation.

    `boto3` est importé paresseusement : une installation qui n'utilise que
    l'ingestion directe n'a aucune raison d'en dépendre, et le service doit
    démarrer sans lui.
    """
    global _client
    if _client is not None:
        return _client

    if not settings.s3_ready:
        raise StockageNonConfigure(
            "stockage objet non configuré : renseignez S3_ENDPOINT, "
            "S3_ACCESS_KEY et S3_SECRET_KEY dans .env"
        )

    try:
        import boto3
        from botocore.config import Config
    except ImportError as e:
        raise StockageNonConfigure(
            "boto3 absent : pip install boto3 (requis pour l'ingestion par pointeur)"
        ) from e

    # MinIO ne sert PAS l'adressage par sous-domaine (bucket.hôte/objet) que les
    # clients S3 utilisent par défaut. Sans `path` ici, chaque téléchargement
    # échoue en résolution DNS, avec un message qui ne désigne pas la cause.
    config = Config(
        signature_version   = "s3v4",
        s3                  = {"addressing_style": "path" if settings.S3_PATH_STYLE_ACCESS else "auto"},
        retries             = {"max_attempts": 3, "mode": "standard"},
        connect_timeout     = 5,
        read_timeout        = 60,
    )
    _client = boto3.client(
        "s3",
        endpoint_url          = settings.S3_ENDPOINT,
        aws_access_key_id     = settings.S3_ACCESS_KEY,
        aws_secret_access_key = settings.S3_SECRET_KEY,
        region_name           = settings.S3_REGION,
        config                = config,
    )
    _log.info("client S3 prêt (%s, path_style=%s)",
              settings.S3_ENDPOINT, settings.S3_PATH_STYLE_ACCESS)
    return _client


# =============================================================================
#  Lecture
# =============================================================================

EXTENSIONS_ACCEPTEES = {".txt", ".md", ".pdf"}

# Types MIME que le backend déclare à l'envoi, vers l'extension correspondante.
TYPES_MIME = {
    "text/plain":              ".txt",
    "text/markdown":           ".md",
    "text/x-markdown":         ".md",
    "application/pdf":         ".pdf",
    "application/x-pdf":       ".pdf",
}


def extension(object_key: str) -> str:
    return os.path.splitext(object_key)[1].lower()


def _format_du_fichier(chemin: str, object_key: str, content_type: str) -> Optional[str]:
    """Détermine le format d'un objet téléchargé : '.txt', '.md', '.pdf', ou None.

    L'extension de la clé ne suffit PAS. Le backend nomme ses objets
    `documents/{organizationId}/{documentId}` — deux UUID, aucune extension.
    S'y fier rejetterait chacun de ses dépôts.

    Trois sources, de la plus fiable à la moins fiable :

      1. Les octets de tête. Un fichier qui commence par `%PDF-` est un PDF,
         quoi qu'en disent la clé et l'en-tête — c'est la seule source qui ne
         peut pas mentir.
      2. L'extension de la clé, quand elle existe et qu'on la connaît.
      3. Le `Content-Type` déclaré au dépôt.

    En dernier recours, un contenu qui se décode en UTF-8 est traité comme du
    texte : mieux vaut indexer un `.log` ou un `.csv` que refuser un document
    valide parce que personne n'a renseigné son type.
    """
    try:
        with open(chemin, "rb") as f:
            tete = f.read(5)
    except OSError:
        tete = b""
    if tete.startswith(b"%PDF-"):
        return ".pdf"

    ext = extension(object_key)
    if ext in EXTENSIONS_ACCEPTEES:
        return ext

    # `Content-Type` peut porter un paramètre : « text/plain; charset=utf-8 ».
    mime = (content_type or "").split(";")[0].strip().lower()
    if mime in TYPES_MIME:
        return TYPES_MIME[mime]

    # Ni signature, ni extension, ni type connu : le contenu décide.
    try:
        with open(chemin, "rb") as f:
            f.read(4096).decode("utf-8")
        return ".txt"
    except (OSError, UnicodeDecodeError):
        return None


def telecharger(bucket: str, object_key: str, dest_dir: str) -> str:
    """Télécharge un objet dans `dest_dir` et retourne le chemin local.

    Le nom local est construit à partir du seul nom de base de la clé : une clé
    contient des séparateurs (`{organizationId}/fichier.pdf`) et, comme tout ce
    qui arrive du réseau sur un service sans authentification, elle ne doit
    jamais être concaténée telle quelle à un chemin de destination.

    Raises:
        ObjetIntrouvable      — l'objet ou le bucket n'existe pas (404)
        ObjetTropVolumineux   — au-delà de S3_MAX_MB (400)
        StockageIndisponible  — MinIO injoignable ou identifiants refusés (502)
    """
    if not bucket or not object_key:
        raise ValueError("bucket et objectKey sont requis")

    client = get_client()

    try:
        from botocore.exceptions import ClientError, BotoCoreError
    except ImportError as e:  # pragma: no cover - get_client a déjà vérifié
        raise StockageIndisponible("boto3 absent") from e

    # Taille connue avant de commencer : inutile de tirer 400 Mo pour les
    # refuser ensuite, et ça donne un message d'erreur exploitable.
    try:
        tete         = client.head_object(Bucket=bucket, Key=object_key)
        taille       = int(tete.get("ContentLength", 0))
        content_type = tete.get("ContentType", "")
    except ClientError as e:
        code = str(e.response.get("Error", {}).get("Code", ""))
        if code in ("404", "NoSuchKey", "NoSuchBucket", "NotFound"):
            raise ObjetIntrouvable(
                f"objet introuvable : {bucket}/{object_key}. "
                f"Vérifiez que le dépôt sur le stockage a bien eu lieu avant la notification."
            ) from None
        raise StockageIndisponible(f"stockage objet : {e}") from e
    except BotoCoreError as e:
        raise StockageIndisponible(f"stockage objet injoignable : {e}") from e

    plafond = settings.S3_MAX_MB * 1024 * 1024
    if taille > plafond:
        raise ObjetTropVolumineux(
            f"objet de {taille / 1048576:.1f} Mo — plafond S3_MAX_MB = {settings.S3_MAX_MB} Mo"
        )

    # Le format ne peut pas être tranché ici : la clé n'a souvent aucune
    # extension, et seul le contenu téléchargé permet de lire sa signature.
    # Le rejet éventuel a donc lieu APRÈS le téléchargement, borné par S3_MAX_MB.
    nom = os.path.basename(object_key) or "document"

    os.makedirs(dest_dir, exist_ok=True)
    # Préfixe unique : deux organisations peuvent déposer le même nom de fichier
    # en même temps, et l'une écraserait le téléchargement de l'autre.
    fd, chemin = tempfile.mkstemp(prefix="s3_", suffix=f"_{nom}", dir=dest_dir)
    os.close(fd)

    try:
        client.download_file(bucket, object_key, chemin)
    except ClientError as e:
        os.unlink(chemin)
        code = str(e.response.get("Error", {}).get("Code", ""))
        if code in ("404", "NoSuchKey", "NoSuchBucket", "NotFound"):
            raise ObjetIntrouvable(f"objet introuvable : {bucket}/{object_key}") from None
        raise StockageIndisponible(f"stockage objet : {e}") from e
    except BotoCoreError as e:
        os.unlink(chemin)
        raise StockageIndisponible(f"stockage objet injoignable : {e}") from e

    fmt = _format_du_fichier(chemin, object_key, content_type)
    if fmt is None:
        os.unlink(chemin)
        raise ValueError(
            f"format non supporté (clé « {object_key} », Content-Type "
            f"« {content_type or 'absent'} ») — accepté : "
            f"{', '.join(sorted(EXTENSIONS_ACCEPTEES))}"
        )

    # L'extraction en aval choisit son lecteur d'après l'extension du chemin :
    # le fichier temporaire doit donc la porter, même quand la clé ne l'a pas.
    if not chemin.endswith(fmt):
        final = chemin + fmt
        os.replace(chemin, final)
        chemin = final

    _log.info("téléchargé %s/%s (%.1f Ko, %s ← %s)", bucket, object_key,
              taille / 1024, fmt, content_type or "type absent")
    return chemin


def sante() -> dict:
    """État du stockage objet, pour /health. N'échoue jamais."""
    if not settings.s3_ready:
        return {"configure": False, "joignable": None}
    try:
        get_client().list_buckets()
        return {"configure": True, "joignable": True}
    except Exception as e:  # noqa: BLE001
        return {"configure": True, "joignable": False, "erreur": str(e)[:200]}
