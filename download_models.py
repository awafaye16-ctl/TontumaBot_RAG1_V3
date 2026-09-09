"""Télécharge les modèles V3 depuis HuggingFace Hub
et les place dans leurs dossiers respectifs sous src/.

Usage :
    python download_models.py

Le modèle STT français (Whisper, cf. STT_FR_MODEL) est chargé depuis le cache
HuggingFace plutôt qu'un dossier dédié : il est récupéré ici pour éviter un
téléchargement au milieu d'une session sur la borne.

Les modèles sont téléchargés une seule fois.
Si le dossier de destination contient déjà des fichiers, le
téléchargement est ignoré (vérification par présence de config.json).
"""
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

BASE = Path(__file__).resolve().parent  # V3/

MODELES = [
    {
        "repo_id":   "soynade-research/Wolof-HuBERT-CTC",
        "local_dir": BASE / "src" / "stt_wolof-hubert-ctc",
        "label":     "STT — soynade-research/Wolof-HuBERT-CTC",
    },
    {
        "repo_id":   os.getenv("STT_FR_MODEL", "openai/whisper-small"),
        "local_dir": None,          # cache HF partagé
        "label":     "STT français — Whisper",
    },
    {
        "repo_id":   "bilalfaye/nllb-200-distilled-600M-wo-fr-en",
        "local_dir": BASE / "src" / "Wo_fr_bilalfayenllb-200-distilled-600M-wo-fr-en",
        "label":     "WO→FR — bilalfaye/nllb-200-distilled-600M-wo-fr-en",
    },
    {
        "repo_id":   "Lahad/nllb200-francais-wolof",
        "local_dir": BASE / "src" / "Fr-Wo_Lahadnllb200-francais-wolof",
        "label":     "FR→WO — Lahad/nllb200-francais-wolof",
    },
]


def already_downloaded(local_dir: Path) -> bool:
    """Vérifie si le dossier contient déjà un modèle (config.json présent)."""
    return (local_dir / "config.json").exists()


def main():
    print("=" * 60)
    print("TontumaBot V3 — Téléchargement des modèles")
    print("=" * 60)

    for m in MODELES:
        local_dir = m["local_dir"]

        # Sans dossier dédié : téléchargement dans le cache HuggingFace
        if local_dir is None:
            print(f"\n⬇️  Téléchargement : {m['label']}")
            print(f"   Hub             : {m['repo_id']}")
            try:
                snapshot_download(repo_id=m["repo_id"])
                print("   ✅ En cache")
            except Exception as e:  # noqa: BLE001
                print(f"   ❌ Échec : {e}")
            continue

        local_dir.mkdir(parents=True, exist_ok=True)

        if already_downloaded(local_dir):
            print(f"\n✅ Déjà présent   : {m['label']}")
            print(f"   Dossier        : {local_dir}")
            continue

        print(f"\n⬇️  Téléchargement : {m['label']}")
        print(f"   Hub             : {m['repo_id']}")
        print(f"   Destination     : {local_dir}")
        try:
            snapshot_download(
                repo_id=m["repo_id"],
                local_dir=str(local_dir),
                ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
            )
            print(f"   ✅ Téléchargé avec succès.")
        except Exception as e:
            print(f"   ❌ Échec : {e}")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("Tous les modèles sont prêts.")
    print("=" * 60)

    # Mise à jour du .env pour pointer vers les dossiers locaux
    env_file = BASE / ".env"
    if not env_file.exists():
        env_example = BASE / ".env.example"
        if env_example.exists():
            import shutil
            shutil.copy(env_example, env_file)
            print(f"\n📄 .env créé depuis .env.example")

    print("\nChemins à utiliser dans .env :")
    for m in MODELES:
        if m["local_dir"] is None:
            continue          # cache HuggingFace, pas de chemin à renseigner
        rel = os.path.relpath(m["local_dir"], BASE)
        print(f"  {rel}")


if __name__ == "__main__":
    main()
