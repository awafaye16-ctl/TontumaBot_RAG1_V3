"""Mémoire conversationnelle — V3.

Chaque session tient un tableau de messages au format usuel :

    messages = [
        {"role": "user",      "content": "Je veux un extrait de naissance."},
        {"role": "assistant", "content": "Présentez-vous à la mairie avec…"},
        {"role": "user",      "content": "Et combien ça coûte ?"},
    ]

Ce tableau sert à deux choses :
  1. il est envoyé au LLM pour qu'il comprenne les questions de suite
     (« et combien ça coûte ? » n'a de sens qu'avec le tour précédent) ;
  2. il permet de compléter la requête de recherche quand la question est
     trop courte pour être cherchée telle quelle.

Quand le tableau atteint MEMORY_MAX_MESSAGES (10 par défaut, soit cinq
échanges), il est réinitialisé : la conversation repart à zéro.

Le contenu est stocké en **français** — la langue pivot du pipeline. Une
question posée en wolof y figure donc traduite, puisque c'est cette version
que le LLM consomme.
"""
import threading
import time

from config import settings

# Purge des sessions inactives : une borne qui tourne des semaines ne doit pas
# accumuler indéfiniment les conversations de ses usagers.
_TTL_SECONDS  = settings.MEMORY_TTL_MINUTES * 60
_MAX_SESSIONS = 500

_store: dict[str, "Conversation"] = {}
_lock = threading.Lock()


class Conversation:
    """Tableau de messages d'une session, avec réinitialisation automatique."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages: list[dict] = []
        self.resets   = 0          # nombre de réinitialisations depuis le début
        self.total    = 0          # messages ajoutés au total (avant remises à zéro)
        self.touched  = time.time()

    # ── Écriture ──────────────────────────────────────────────────────────
    def add(self, role: str, content: str) -> bool:
        """Ajoute un message. Retourne True si la mémoire vient d'être vidée.

        La limite porte sur le nombre d'entrées du tableau, tous rôles
        confondus : dix messages = cinq échanges.
        """
        content = (content or "").strip()
        self.touched = time.time()
        if not content:
            return False

        self.messages.append({"role": role, "content": content})
        self.total += 1

        if len(self.messages) >= settings.MEMORY_MAX_MESSAGES:
            self.reset()
            return True
        return False

    def add_exchange(self, question: str, answer: str) -> bool:
        """Ajoute la paire question/réponse d'un tour. True si vidée au passage."""
        vide = self.add("user", question)
        return self.add("assistant", answer) or vide

    def reset(self) -> None:
        self.messages = []
        self.resets  += 1
        self.touched  = time.time()

    # ── Lecture ───────────────────────────────────────────────────────────
    def history(self) -> list[dict]:
        """Copie du tableau — l'appelant ne doit pas muter la mémoire."""
        self.touched = time.time()
        return [dict(m) for m in self.messages]

    def last_user_message(self) -> str | None:
        for m in reversed(self.messages):
            if m["role"] == "user":
                return m["content"]
        return None

    def snapshot(self) -> dict:
        return {
            "session_id": self.session_id,
            "messages":   self.history(),
            "count":      len(self.messages),
            "max":        settings.MEMORY_MAX_MESSAGES,
            "resets":     self.resets,
            "total":      self.total,
        }


# =============================================================================
#  Magasin de sessions
# =============================================================================

def _purge_locked() -> None:
    """Retire les sessions inactives, puis les plus anciennes si trop nombreuses."""
    limite = time.time() - _TTL_SECONDS
    for sid in [s for s, c in _store.items() if c.touched < limite]:
        _store.pop(sid, None)

    if len(_store) > _MAX_SESSIONS:
        anciennes = sorted(_store.items(), key=lambda kv: kv[1].touched)
        for sid, _ in anciennes[: len(_store) - _MAX_SESSIONS]:
            _store.pop(sid, None)


def get(session_id: str | None) -> Conversation | None:
    """Conversation de cette session, créée au besoin. None si mémoire coupée."""
    if not session_id or not settings.MEMORY_ENABLED:
        return None
    with _lock:
        conv = _store.get(session_id)
        if conv is None:
            _purge_locked()
            conv = _store[session_id] = Conversation(session_id)
        return conv


def reset(session_id: str) -> bool:
    """Vide la mémoire d'une session. False si elle était inconnue."""
    with _lock:
        conv = _store.get(session_id)
    if conv is None:
        return False
    conv.reset()
    return True


def forget(session_id: str) -> bool:
    """Supprime complètement la session (fin de session sur la borne)."""
    with _lock:
        return _store.pop(session_id, None) is not None


def snapshot(session_id: str) -> dict | None:
    with _lock:
        conv = _store.get(session_id)
    return conv.snapshot() if conv else None


def sessions() -> int:
    with _lock:
        return len(_store)
