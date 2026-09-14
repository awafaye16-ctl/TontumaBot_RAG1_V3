"""TTS V3 — Oolel-Voices uniquement.

Modèle : soynade-research/Oolel-Voices (voice cloning, meilleure qualité)
Voice prompt : fichier audio de référence pour l'identité vocale.
"""
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import settings  # noqa: E402

import numpy as np
import soundfile as sf

from journal import journal

_log = journal("tts")

# Énoncé de préchauffage : longueur représentative d'un morceau réel, pour que
# les formes de tenseur compilées au démarrage servent aux vraies requêtes.
_TEXTE_PRECHAUFFAGE = (
    "Jàmm nga fanaan. Soo bëggee dem ci fajkatu fajukaay bi ngir faju, danga "
    "war a jaar ci buntu bu mag bi, ba noppi nga dem ci batimaa A fa nekk "
    "konsil yi."
)

# ── État ───────────────────────────────────────────────────────────────────
_model  = None
_ready  = False
_prompt = None


def _find_cache() -> Path | None:
    """Cherche le snapshot Oolel-Voices dans le cache HuggingFace local."""
    cache = (
        Path.home()
        / ".cache" / "huggingface" / "hub"
        / "models--soynade-research--Oolel-Voices"
        / "snapshots"
    )
    if cache.exists():
        dirs = sorted(cache.iterdir())
        if dirs:
            return dirs[-1]
    return None


def _accelerer_attention(model) -> bool:
    """Rend au décodeur T3 son noyau d'attention optimisé (SDPA).

    Sa boucle de génération réclame `output_attentions=True` à chaque pas
    (models/t3/t3.py). Les poids d'attention ne servent qu'à
    l'AlignmentStreamAnalyzer, garde-fou anti-hallucination que le modèle ne
    construit que pour les checkpoints multilingues — et celui-ci ne l'est pas
    (`text_tokens_dict_size=704`). Les matrices sont donc calculées puis
    jetées, mais leur seule demande suffit à faire retomber Llama de SDPA sur
    l'attention manuelle, qui matérialise la carte complète à chaque pas.

    Mesuré (M1, MPS, n_cfm=4, graine fixe) : 18,6 s → 10,8 s sur une phrase de
    85 caractères, pour un audio identique à l'échantillon près — ce n'est pas
    un compromis de qualité, c'est du calcul rendu.

    Retourne False sans rien toucher si le checkpoint est multilingue : là,
    l'analyzer lit vraiment les attentions et les couper le casserait.
    """
    if getattr(model.t3.hp, "is_multilingual", False):
        _log.info("checkpoint multilingue : attentions conservées "
                  "(AlignmentStreamAnalyzer actif)")
        return False

    from models.t3.inference.t3_hf_backend import T3HuggingfaceBackend

    if getattr(T3HuggingfaceBackend.forward, "_sans_attentions", False):
        return True                      # déjà patché (rechargement du module)

    _forward_origine = T3HuggingfaceBackend.forward

    def forward(self, *args, **kwargs):
        kwargs["output_attentions"] = False
        return _forward_origine(self, *args, **kwargs)

    forward._sans_attentions = True
    T3HuggingfaceBackend.forward = forward
    return True


def _load() -> bool:
    """Charge Oolel-Voices depuis le Hub (ou le cache). Retourne True si succès."""
    global _model, _ready, _prompt
    if _ready:
        return True
    try:
        ckpt_dir = _find_cache()
        if ckpt_dir is None:
            _log.info(f"Téléchargement Oolel-Voices ({settings.OOLEL_TTS_REPO})...")
            from huggingface_hub import snapshot_download
            ckpt_dir = Path(snapshot_download(repo_id=settings.OOLEL_TTS_REPO))
            _log.info(f"Oolel-Voices téléchargé → {ckpt_dir}")
        ckpt_str = str(ckpt_dir)
        if ckpt_str not in sys.path:
            sys.path.insert(0, ckpt_str)

        _log.info(f"Chargement Oolel-Voices depuis {ckpt_dir}...")
        from modeling_oolel_voices import OolelVoicesForInference  # type: ignore

        # Backend commun au pipeline (surcharge : TTS_DEVICE)
        import device as _dev
        _device = _dev.resolve("tts")

        _model = OolelVoicesForInference.from_pretrained(ckpt_str, device_map=_device)
        _model.eval()

        if _accelerer_attention(_model):
            _log.info("Attention optimisée (SDPA) rétablie sur le décodeur T3")
        # ── Accélération vocodeur : réduire les itérations de diffusion ──
        n_cfm = getattr(settings, "TTS_N_STEPS", 10)
        if n_cfm and 1 <= n_cfm < 10:
            s3gen     = _model.s3gen
            _orig_inf = s3gen.inference

            def _fast_inference(*args, n_cfm_timesteps=None, **kwargs):
                if n_cfm_timesteps is None:
                    n_cfm_timesteps = n_cfm
                return _orig_inf(*args, n_cfm_timesteps=n_cfm_timesteps, **kwargs)

            s3gen.inference = _fast_inference
            _log.info(f"Vocodeur accéléré : {n_cfm} itérations (au lieu de 10)")
        # Prompt audio de référence pour le voice cloning
        candidates = [
            ckpt_dir / "8_1_c.wav",
            settings.BASE_DIR / "8_1_c.wav",
            settings.BASE_DIR.parent / "Oolel-Voices" / "8_1_c.wav",
        ]
        for c in candidates:
            if c.exists():
                _prompt = str(c)
                _log.info(f"Voice prompt : {_prompt}")
                break

        if _prompt is None:
            _log.warning("⚠️  Aucun fichier audio de référence trouvé — synthèse sans voice prompt.")
        else:
            # L'identité vocale est extraite une fois pour toutes. Passer
            # `audio_prompt_path` à generate() relancerait sinon le décodage du
            # fichier de référence (36 s d'audio) et l'encodeur de voix à
            # chaque appel — et le découpage en phrases multiplie les appels.
            t0 = time.perf_counter()
            _model.prepare_conditionals(_prompt, exaggeration=0.5)
            _log.info(f"Identité vocale préparée ({time.perf_counter() - t0:.1f}s)")
        _ready = True
        _log.info("Oolel-Voices prêt.")
        return True

    except Exception as e:
        _log.error(f"Oolel-Voices indisponible ({type(e).__name__}: {e})")
        return False


# =============================================================================
#  Découpage en morceaux
# =============================================================================
#  Le décodeur T3 est auto-régressif : il génère les jetons de parole un par
#  un, en gardant tout le contexte déjà produit. Une réponse entière — 332
#  caractères en moyenne sur les 20 questions de référence, jusqu'à 903 — tient
#  donc la mémoire du début à la fin, sur une machine qui n'en a plus (mesuré :
#  9,3 Go de swap sur 10,2). C'est là que les temps deviennent erratiques : la
#  même phrase, à graine fixée et audio identique, a mis 153 s puis 44 s.
#
#  Découpé, chaque passe reste bornée et le premier son arrive sans attendre la
#  fin (15,4 s au lieu de 43,6 s sur une réponse de trois phrases). Le total,
#  lui, ne bouge pas : une fois l'attention optimisée rétablie, le coût est
#  proportionnel à la durée d'audio, pas au carré. Le gain de découper est donc
#  la stabilité et le temps jusqu'au premier son, pas le débit.
#
#  On coupe aux fins de phrase, jamais au milieu : la prosodie d'Oolel-Voices
#  se construit sur l'énoncé entier, et un morceau tronqué s'entend. Les
#  phrases courtes sont regroupées jusqu'à la longueur visée pour la même
#  raison — trois mots synthétisés seuls sonnent hachés.

_FIN_DE_PHRASE = re.compile(r"(?<=[.!?…])\s+|\n+")

# Une phrase qui dépasse à elle seule la cible est recoupée ici, faute de mieux.
_RESPIRATION = re.compile(r"(?<=[,;:])\s+")

# En deçà, un morceau ne s'entend plus comme une phrase mais comme un bout :
# il est recollé au précédent même si la cible est dépassée.
_MORCEAU_MINIMAL = 40


def _decouper(texte: str, cible: int) -> list[str]:
    """Découpe `texte` en morceaux d'environ `cible` caractères.

    Les coupures tombent aux fins de phrase ; une phrase plus longue que la
    cible est recoupée aux virgules, et en dernier recours laissée entière —
    mieux vaut un morceau trop long qu'un énoncé coupé au milieu d'un mot.
    """
    if cible <= 0:
        return [texte] if texte.strip() else []

    phrases: list[str] = []
    for phrase in _FIN_DE_PHRASE.split(texte):
        phrase = (phrase or "").strip()
        if not phrase:
            continue
        if len(phrase) <= cible:
            phrases.append(phrase)
            continue
        # Trop longue : on cherche des respirations à l'intérieur.
        courant = ""
        for bout in _RESPIRATION.split(phrase):
            if courant and len(courant) + len(bout) + 1 > cible:
                phrases.append(courant)
                courant = bout
            else:
                courant = f"{courant} {bout}".strip()
        if courant:
            phrases.append(courant)

    morceaux: list[str] = []
    for phrase in phrases:
        # Une phrase rejoint la précédente si le morceau tient encore dans la
        # cible — ou, quand l'un des deux est trop court pour s'entendre seul,
        # en la dépassant d'au plus un morceau minimal. Sans cette borne, une
        # liste numérotée (« 1. Wone sa kayit. 2. Xaar... ») recollerait tous
        # ses éléments courts les uns aux autres et reformerait un bloc.
        if morceaux:
            fusion = len(morceaux[-1]) + len(phrase) + 1
            trop_court = (len(morceaux[-1]) < _MORCEAU_MINIMAL
                          or len(phrase) < _MORCEAU_MINIMAL)
            if fusion <= cible or (trop_court and fusion <= cible + _MORCEAU_MINIMAL):
                morceaux[-1] = f"{morceaux[-1]} {phrase}"
                continue
        morceaux.append(phrase)

    return morceaux


def _synthetiser(texte: str) -> np.ndarray:
    """Un appel au modèle, sur un morceau déjà borné."""
    wav = _model.generate(
        texte,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
    )
    return wav.squeeze(0).detach().cpu().numpy()


def _etirer(audio: np.ndarray) -> np.ndarray:
    """Applique TTS_SPEED (time-stretch, préserve la hauteur)."""
    speed = getattr(settings, "TTS_SPEED", 1.0)
    if not speed or speed <= 0 or abs(speed - 1.0) <= 1e-3:
        return audio
    import librosa
    t0 = time.perf_counter()
    # librosa.effects.time_stretch(rate>1) → ralentit
    etire = librosa.effects.time_stretch(audio, rate=speed)
    _log.info(f"Time-stretch rate={speed} ({time.perf_counter() - t0:.2f}s)")
    return etire


def _chemin_morceau(out_path: str, index: int) -> str:
    """`.../response_ab12cd34.wav` → `.../response_ab12cd34_01.wav`."""
    base, ext = os.path.splitext(out_path)
    return f"{base}_{index:02d}{ext}"


def synthesize(text: str, out_path: str, on_chunk=None) -> str:
    """Synthétise le texte avec Oolel-Voices et sauvegarde dans out_path.

    `on_chunk(index, total, chemin)` — optionnel — est appelé dès qu'un morceau
    est prêt, avec son propre fichier audio. C'est ce qui permet à la borne de
    commencer à parler pendant que la suite se synthétise : sur une réponse de
    trois phrases, le premier son tombe à 15 s au lieu d'attendre les 44 s de
    l'énoncé complet. Le fichier `out_path` est écrit dans tous les cas, avec
    la réponse entière — les consommateurs de l'API qui ignorent les morceaux
    continuent de recevoir un seul WAV.
    """
    if not _load():
        raise RuntimeError("Oolel-Voices impossible à charger")

    morceaux = _decouper(text, getattr(settings, "TTS_CHUNK_CHARS", 200))
    if not morceaux:
        raise ValueError("Rien à synthétiser : texte vide")
    total = len(morceaux)

    silence = np.zeros(
        int(_model.sr * getattr(settings, "TTS_CHUNK_PAUSE_MS", 120) / 1000),
        dtype=np.float32,
    )

    segments: list[np.ndarray] = []
    t_debut = time.perf_counter()
    for i, morceau in enumerate(morceaux, 1):
        t0 = time.perf_counter()
        # L'étirement est appliqué morceau par morceau : le fichier diffusé et
        # le fichier complet doivent s'entendre pareil.
        audio = _etirer(_synthetiser(morceau))
        if total > 1:
            _log.info("morceau %d/%d (%d car.) → %.1f s d'audio en %.1f s",
                      i, total, len(morceau), len(audio) / _model.sr,
                      time.perf_counter() - t0)
            if segments:
                segments.append(silence)
        segments.append(audio)

        if on_chunk is not None:
            # Un morceau seul : son fichier est le fichier final, inutile de
            # l'écrire deux fois.
            chemin = out_path if total == 1 else _chemin_morceau(out_path, i)
            try:
                sf.write(chemin, audio, _model.sr, format="WAV")
                on_chunk(i, total, chemin)
            except Exception as e:
                # La diffusion est un confort : si elle échoue, la synthèse va
                # au bout et le client lira le fichier complet.
                _log.error("diffusion du morceau %d impossible (%s: %s)",
                           i, type(e).__name__, e)

    audio_np = segments[0] if len(segments) == 1 else np.concatenate(segments)
    if total > 1:
        _log.info("%d morceaux → %.1f s d'audio en %.1f s", total,
                  len(audio_np) / _model.sr, time.perf_counter() - t_debut)

    sf.write(out_path, audio_np, _model.sr, format="WAV")
    _rendre_cache()
    return out_path


def _rendre_cache() -> None:
    """Rend au système le cache d'allocateur laissé par la synthèse.

    La synthèse est de loin le plus gros consommateur du pipeline : mesuré,
    elle laisse 1,5 à 1,8 Go de blocs mis en cache derrière elle. Ils seront
    repris à la synthèse suivante — l'appel ne fait pas gagner de mémoire en
    régime établi — mais entre deux usagers d'une borne, qui sont séparés par
    des minutes, cette place revient au reste de la machine.

    Sans effet hors MPS : CUDA a sa propre gestion et le CPU n'a pas ce cache.
    """
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass          # une libération de cache qui échoue ne doit rien casser


def prechauffer() -> bool:
    """Synthèse à vide, pour absorber la montée en température du moteur.

    Les noyaux Metal se compilent au premier passage et cette compilation se
    paie une fois par processus. Mesuré sur une réponse réelle de 357
    caractères, sans préchauffage : RTF 3,19 à la première requête contre 1,99
    à la seconde. Avec, la première tombe à 2,23.

    Le coût (≈65 s sur M1) est déplacé, pas supprimé : cette fonction n'a de
    sens qu'au démarrage, appelée par le warm-up de l'application, où personne
    n'attend devant la borne. En chargement paresseux (`WARMUP_TTS=false`) elle
    n'est pas appelée — la payer dans la première requête serait pire que tout.

    L'énoncé fait la longueur d'un morceau réel : une phrase de trois mots ne
    compile pas les mêmes formes de tenseur et ne préchauffe presque rien
    (RTF 3,19 à la première requête, soit le niveau sans préchauffage).
    """
    if not _load():
        return False
    t0 = time.perf_counter()
    try:
        _synthetiser(_TEXTE_PRECHAUFFAGE)
    except Exception as e:
        # Un préchauffage raté ne doit pas condamner le TTS : la vraie synthèse
        # retentera, et échouera alors avec son propre message.
        _log.warning(f"Préchauffage ignoré ({type(e).__name__}: {e})")
        return False
    _log.info(f"Moteur préchauffé ({time.perf_counter() - t0:.1f}s)")
    return True


def source() -> str:
    """Retourne le nom du moteur TTS (pour les traces)."""
    return "oolel-voices" if _ready else "indisponible"


def device() -> str:
    """Backend effectivement utilisé — pour /health et les traces."""
    import device as _dev
    return _dev.resolve("tts", verbeux=False)


if __name__ == "__main__":
    import sys as _sys
    txt = _sys.argv[1] if len(_sys.argv) > 1 else "Jàmm nga fanaan. Nanga def?"
    out = _sys.argv[2] if len(_sys.argv) > 2 else "test_v3.wav"
    result = synthesize(txt, out)
    print(f"{source()} → {result}")