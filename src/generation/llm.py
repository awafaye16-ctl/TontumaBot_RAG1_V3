"""Générateur de réponse LLM — V3.

Providers disponibles :
  groq   → Groq API  (openai/gpt-oss-120b par défaut, rapide). Ce modèle
            raisonne avant de répondre : sa trace interne (~200 jetons) est
            comptée dans max_tokens, d'où un plafond large — cf. LLM_MAX_TOKENS.
  gemini → Google Gemini API (gemini-2.0-flash)
  local  → Qwen2.5-7B-Instruct en 4bit NF4 (retenu après benchmark,
            ~4.7 Go VRAM, tient sur T4)
  fallback → renvoie le premier passage du contexte si aucune clé API

Prompt système : réponse strictement à partir du contexte,
abstention explicite si l'information est absente.
"""
import re
from config import settings

# La réponse est lue à voix haute par la borne. Mesuré : la synthèse coûte
# environ deux secondes par seconde d'audio, soit ~0,14 s par caractère — une
# réponse de 900 caractères fait attendre deux minutes. La consigne de longueur
# est donc une contrainte de latence, pas une préférence de style, et c'est elle
# qui règle la longueur : le plafond de jetons n'est qu'un garde-fou.
_BREVETE = (
    f"Réponds en {settings.LLM_MAX_PHRASES} phrases au maximum. "
    "Ta réponse est lue à voix haute devant l'usager : chaque phrase de plus "
    "le fait attendre plusieurs secondes. Donne ce dont il a besoin pour agir "
    "— le lieu, la démarche, le document, le montant — et laisse de côté les "
    "précisions accessoires, les rappels de politesse et les reformulations de "
    "la question. Si la démarche comporte des étapes, n'en garde que les "
    "principales, une phrase chacune. "
    # Mesuré : sommé d'être bref, le modèle comprime en devenant vague — « le
    # délai varie selon l'urgence » remplaçait « 48 à 72 heures ». Une réponse
    # courte et floue est pire qu'une réponse longue : l'usager repart sans
    # savoir quoi faire, et repose la question.
    "Raccourcis en supprimant des points, jamais en remplaçant un chiffre, un "
    "lieu, un délai ou un montant précis par une formule générale : mieux vaut "
    "dire moins de choses que les dire vaguement. "
)

SYSTEM_PROMPT = (
    "Tu es TontumaBot, l'assistant administratif officiel du Sénégal. "
    "Réponds en français, de manière concise et directe. " + _BREVETE +
    "Utilise en priorité les passages fournis dans le contexte de façon reformulée. "
    "Si les passages contiennent une information partiellement liée, utilise-la pour formuler une réponse utile. "
    "Si et seulement si le contexte ne contient AUCUNE information pertinente sur la question, "
    "réponds : 'Je n'ai pas trouvé cette information dans ma base documentaire.' "
    "Ne jamais inventer de chiffres, adresses ou délais non présents dans les passages. "
    "Ne jamais inclure de balises <think> ou de raisonnement interne dans ta réponse."
    "Les réponses doivent etre comme si elles étaient écrites par un humain, avec des phrases complètes et un style naturel."
    "L'historique de conversation ne sert qu'à comprendre la question courante "
    "(pronoms, sous-entendus, questions de suite) : il n'est jamais une source d'information. "
    "Toute donnée factuelle doit venir des passages du contexte."
)

SYSTEM_PROMPT_PLAIN = (
    "Tu es TontumaBot, l'assistant administratif officiel du Sénégal. "
    "Réponds en français, de manière concise et directe. " + _BREVETE +
    "N'utilise AUCUN formatage markdown (pas de **, pas de #, pas de listes à puces *). "
    "Écris en prose simple avec des phrases complètes et des numéros (1. 2. 3.) si besoin. "
    "Utilise en priorité les passages fournis dans le contexte de façon reformulée. "
    "Si les passages contiennent une information partiellement liée, utilise-la pour formuler une réponse utile. "
    "Si et seulement si le contexte ne contient AUCUNE information pertinente sur la question, "
    "réponds : 'Je n'ai pas trouvé cette information dans ma base documentaire.' "
    "Ne jamais inventer de chiffres, adresses ou délais non présents dans les passages. "
    # Cette réponse part vers NLLB : mesuré, « cinq mille francs » ressort en
    # wolof à « junni » (mille), tandis que « 5000 francs » traverse intact.
    # Les nombres en chiffres sont donc une contrainte de sûreté, pas un style.
    "Écris TOUJOURS les nombres en chiffres (15, 1000, 5000), jamais en toutes "
    "lettres : montants, délais, quantités, étages, numéros. "
    "Ne jamais inclure de balises <think> ou de raisonnement interne dans ta réponse."
    "Les réponses doivent etre comme si elles étaient écrites par un humain, avec des phrases complètes et un style naturel."
    "L'historique de conversation ne sert qu'à comprendre la question courante "
    "(pronoms, sous-entendus, questions de suite) : il n'est jamais une source d'information. "
    "Toute donnée factuelle doit venir des passages du contexte."
)

def _finir_sur_phrase(texte: str) -> str:
    """Retire une dernière phrase laissée en plan par le plafond de jetons.

    Une réponse tronquée n'est pas seulement inélégante : elle part telle
    quelle à la synthèse vocale, et la borne s'arrête au milieu d'un mot. Mieux
    vaut rendre une phrase de moins qu'une phrase coupée.

    Si le texte ne contient aucune fin de phrase — cas d'une réponse d'un seul
    tenant très longue — on le rend inchangé : le tronquer davantage ne
    l'améliorerait pas.
    """
    fins = [texte.rfind(c) for c in ".!?…"]
    coupe = max(fins)
    return texte[:coupe + 1].rstrip() if coupe > 0 else texte


def _clean_history(history) -> list[dict]:
    """Ne garde que des entrées {role, content} exploitables, en français."""
    propre = []
    for m in history or []:
        role    = (m.get("role") or "").strip()
        content = (m.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            propre.append({"role": role, "content": content})
    return propre


def _history_transcript(history) -> str:
    """Rend l'historique en texte, pour les providers sans API de messages."""
    lignes = [("Usager : " if m["role"] == "user" else "Assistant : ") + m["content"]
              for m in _clean_history(history)]
    return "\n".join(lignes)


# ── État modèle local ─────────────────────────────────────────────────────
_local_model     = None
_local_tokenizer = None
_local_ready     = False


# ─────────────────────────────────────────────────────────────────────────────

def _groq_generate(question_fr: str, context_fr: str, plain: bool = False,
                   history=None) -> str:
    from groq import Groq
    client = Groq(api_key=settings.GROQ_API_KEY)
    prompt = SYSTEM_PROMPT_PLAIN if plain else SYSTEM_PROMPT
    # L'historique passe comme de vrais tours de dialogue, entre la consigne
    # système et la question courante : c'est la forme que le modèle attend.
    completion = client.chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            *_clean_history(history),
            {"role": "user",   "content": f"Contexte :\n{context_fr}\n\nQuestion : {question_fr}"},
        ],
        temperature=0.2,
        max_tokens=settings.LLM_MAX_TOKENS,
    )
    choix = completion.choices[0]
    raw   = choix.message.content or ""
    # _strip_think gère les balises <think>...</think> complètes ET incomplètes
    texte = _strip_think(raw)

    if getattr(choix, "finish_reason", None) == "length":
        print(f"[llm] réponse coupée au plafond de {settings.LLM_MAX_TOKENS} jetons")
        texte = _finir_sur_phrase(texte)

    # Un modèle à raisonnement peut consommer tout le budget avant d'écrire un
    # mot : le contenu revient alors vide. Rendre une réponse vide serait le
    # pire des cas — la borne se tairait sans rien afficher. On le signale, et
    # on sert le passage le plus pertinent plutôt que le silence.
    if not texte.strip():
        print(f"[llm] réponse vide — le plafond de {settings.LLM_MAX_TOKENS} jetons "
              f"est trop bas pour ce modèle (sa trace de raisonnement y est "
              f"comptée). Repli sur le contexte.")
        return _fallback(question_fr, context_fr)
    return texte


def _gemini_generate(question_fr: str, context_fr: str, plain: bool = False,
                     history=None) -> str:
    import google.generativeai as genai
    genai.configure(api_key=settings.GEMINI_API_KEY)
    model  = genai.GenerativeModel(settings.GEMINI_MODEL)
    prompt = SYSTEM_PROMPT_PLAIN if plain else SYSTEM_PROMPT
    passe  = _history_transcript(history)
    bloc   = f"Conversation précédente :\n{passe}\n\n" if passe else ""
    full   = f"{prompt}\n\n{bloc}Contexte :\n{context_fr}\n\nQuestion : {question_fr}"
    reponse = model.generate_content(
        full,
        generation_config={"max_output_tokens": settings.LLM_MAX_TOKENS},
    )
    texte = reponse.text
    # Gemini expose la raison d'arrêt par candidat ; MAX_TOKENS vaut 2.
    try:
        if reponse.candidates and int(reponse.candidates[0].finish_reason) == 2:
            print("[llm] réponse coupée au plafond de jetons — dernière phrase retirée")
            texte = _finir_sur_phrase(texte)
    except Exception:
        pass          # une raison d'arrêt illisible ne doit pas perdre la réponse
    return texte


def _load_local_model():
    """Charge le LLM local sur le backend disponible.

    La quantification bitsandbytes (4bit/8bit) suppose des noyaux CUDA : elle
    n'existe ni sur MPS ni sur CPU. On l'ignore donc hors CUDA plutôt que de
    laisser `from_pretrained` échouer sur une erreur obscure — mais un modèle
    7B en fp32 demande ~28 Go, ce qui ne tient pas sur la plupart des machines
    sans GPU. Le message le dit franchement : hors CUDA, `LLM_PROVIDER=local`
    est un mode de dépannage, pas un mode de production.
    """
    global _local_model, _local_tokenizer, _local_ready
    if _local_ready:
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import device as _dev

    model_id = settings.LOCAL_LLM_MODEL   # "Qwen/Qwen2.5-7B-Instruct"
    quant    = settings.LOCAL_LLM_QUANT   # "4bit"
    backend  = _dev.resolve("llm")

    kwargs = {}
    if backend == "cuda":
        if quant == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif quant == "8bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            kwargs["torch_dtype"] = torch.float16
        # device_map="auto" répartit les couches (accelerate) : utile en multi-GPU
        # comme en délestage CPU quand la VRAM ne suffit pas.
        kwargs["device_map"] = "auto"
        print(f"[LLM] Chargement local ({model_id}, {quant}) sur CUDA...")
    else:
        if quant in ("4bit", "8bit"):
            print(f"[LLM] {quant} ignoré : bitsandbytes exige CUDA, "
                  f"backend détecté = {backend.upper()}.")
        # fp16 sur MPS (Metal gère la demi-précision et divise la mémoire par
        # deux) ; fp32 sur CPU, où le fp16 est émulé donc plus lent.
        kwargs["torch_dtype"] = torch.float16 if backend == "mps" else torch.float32
        kwargs["device_map"]  = backend
        print(f"[LLM] Chargement local ({model_id}, sans quantification) "
              f"sur {backend.upper()} — prévoir beaucoup de RAM.")

    _local_tokenizer = AutoTokenizer.from_pretrained(model_id)
    _local_model     = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    _local_ready = True
    print("[LLM] Modèle local prêt.")


def _local_generate(question_fr: str, context_fr: str, plain: bool = False,
                    history=None) -> str:
    import torch
    _load_local_model()
    consigne = SYSTEM_PROMPT_PLAIN if plain else SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": consigne},
        *_clean_history(history),
        {"role": "user",   "content": f"Contexte :\n{context_fr}\n\nQuestion : {question_fr}"},
    ]
    inputs   = _local_tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    ).to(_local_model.device)
    with torch.no_grad():
        output = _local_model.generate(
            inputs,
            max_new_tokens=settings.LLM_MAX_TOKENS,
            do_sample=False,
            pad_token_id=_local_tokenizer.eos_token_id,
        )
    raw = _local_tokenizer.decode(output[0][inputs.shape[-1]:], skip_special_tokens=True)
    return _strip_think(raw.strip())


def _strip_think(text: str) -> str:
    """Supprime les balises <think>...</think> (Qwen, DeepSeek…).
    Gère aussi les balises ouvertes non fermées.
    """
    # Cas 1 : balise complète <think>...</think>
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    # Cas 2 : balise ouverte non fermée <think>... (tout ce qui suit)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text.strip()


def _fallback(question_fr: str, context_fr: str) -> str:
    """Retourne le premier passage du contexte si aucun LLM disponible."""
    if not context_fr.strip():
        return "Je n'ai pas trouvé d'information dans la base documentaire."
    return context_fr.split("\n\n")[0]

# ─────────────────────────────────────────────────────────────────────────────

def generate(question_fr: str, context_fr: str, provider: str = "groq",
             plain: bool = False, history=None) -> str:
    """Génère la réponse FR à partir du contexte.

    provider : 'groq' | 'gemini' | 'local' | autre → fallback
    plain    : True → prose sans markdown (pour traduction NLLB vers wolof)
    history  : tours précédents [{role, content}] en français, pour résoudre
               les questions de suite. Sans effet sur les faits cités.
    """
    if provider == "gemini":
        if settings.GEMINI_API_KEY:
            return _gemini_generate(question_fr, context_fr, plain=plain, history=history)
        return _fallback(question_fr, context_fr)

    if provider == "local":
        return _local_generate(question_fr, context_fr, plain=plain, history=history)

    # Groq (défaut)
    if settings.GROQ_API_KEY:
        return _groq_generate(question_fr, context_fr, plain=plain, history=history)

    return _fallback(question_fr, context_fr)
