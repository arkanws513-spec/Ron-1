import hashlib
import threading
from pathlib import Path

from core.config import settings
from core.memory import Memory
from core.reasoning import build_messages
from model.qwen_backend import QwenBackend


class Ron:
    def __init__(self):
        self.backend = QwenBackend(settings.model_id)
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()
        self._generation_lock = threading.Lock()
        self._memory_root = Path(settings.memory_path)

    def _conversation_lock(self, key: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(key, threading.Lock())

    def _memory_for(self, user_id: str, conversation_id: str) -> Memory:
        # Hash identifiers before using them as path components. User IDs and
        # conversation IDs never become raw filesystem paths.
        user_key = hashlib.sha256(user_id.encode("utf-8")).hexdigest()
        conversation_key = hashlib.sha256(conversation_id.encode("utf-8")).hexdigest()
        return Memory(str(self._memory_root / user_key / f"{conversation_key}.json"))

    def chat(
        self,
        user_message: str,
        user_id: str = "local",
        conversation_id: str = "default",
    ) -> str:
        memory = self._memory_for(user_id, conversation_id)
        lock_key = hashlib.sha256(
            f"{user_id}\0{conversation_id}".encode("utf-8")
        ).hexdigest()

        # Prevent lost updates within a conversation and avoid concurrent model
        # generation from exhausting RAM or touching model state concurrently.
        with self._conversation_lock(lock_key):
            history = memory.load()
            messages = build_messages(history, user_message)
            with self._generation_lock:
                answer = self.backend.generate(
                    messages,
                    max_new_tokens=settings.max_new_tokens,
                    temperature=settings.temperature,
                    top_p=settings.top_p,
                )
            memory.save(
                history
                + [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": answer},
                ]
            )
            return answer
