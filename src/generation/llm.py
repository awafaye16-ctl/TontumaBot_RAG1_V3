"""Générateur de réponse LLM — V3.

Providers disponibles :
  groq   → Groq API  (openai/gpt-oss-120b par défaut, non-reasoning, rapide)
  gemini → Google Gemini API (gemini-2.0-flash)
  local  → Qwen2.5-7B-Instruct en 4bit NF4 (retenu après benchmark,
            ~4.7 Go VRAM, tient sur T4)
  fallback → renvoie le premier passage du contexte si aucune clé API

Prompt système : réponse strictement à partir du contexte,
abstention explicite si l'information est absente.
"""
import re
from config import settings

SYSTEM_PROMPT = (
    "Tu es TontumaBot, l'assistant administratif officiel du Sénégal. "
    "Réponds en français, de manière concise et directe. "
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
    "Réponds en français, de manière concise et directe. "
    "N'utilise AUCUN formatage markdown (pas de **, pas de #, pas de listes à puces *). "
    "Écris en prose simple avec des phrases complètes et des numéros (1. 2. 3.) si besoin. "
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
    )
    raw = completion.choices[0].message.content or ""
    # _strip_think gère les balises <think>...</think> complètes ET incomplètes
    return _strip_think(raw)


def _gemini_generate(question_fr: str, context_fr: str, plain: bool = False,
                     history=None) -> str:
    import google.generativeai as genai
    genai.configure(api_key=settings.GEMINI_API_KEY)
    model  = genai.GenerativeModel(settings.GEMINI_MODEL)
    prompt = SYSTEM_PROMPT_PLAIN if plain else SYSTEM_PROMPT
    passe  = _history_transcript(history)
    bloc   = f"Conversation précédente :\n{passe}\n\n" if passe else ""
    full   = f"{prompt}\n\n{bloc}Contexte :\n{context_fr}\n\nQuestion : {question_fr}"
    return model.generate_content(full).text


def _load_local_model():
    """Charge Qwen2.5-7B-Instruct en 4bit NF4 depuis HuggingFace Hub."""
    global _local_model, _local_tokenizer, _local_ready
    if _local_ready:
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_id = settings.LOCAL_LLM_MODEL   # "Qwen/Qwen2.5-7B-Instruct"
    quant    = settings.LOCAL_LLM_QUANT   # "4bit"
    print(f"[LLM] Chargement local ({model_id}, {quant})...")

    kwargs = {}
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

    _local_tokenizer = AutoTokenizer.from_pretrained(model_id)
    _local_model     = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", **kwargs
    )
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
            max_new_tokens=220,
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
