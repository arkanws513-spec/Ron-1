from core.config import settings
from core.memory import Memory
from core.reasoning import build_messages
from model.qwen_backend import QwenBackend


class Ron:
    def __init__(self):
        self.memory = Memory(settings.memory_path)
        self.backend = QwenBackend(settings.model_id)

    def chat(self, user_message: str) -> str:
        history = self.memory.load()
        messages = build_messages(history, user_message)
        answer = self.backend.generate(
            messages,
            max_new_tokens=settings.max_new_tokens,
            temperature=settings.temperature,
            top_p=settings.top_p,
        )
        self.memory.append("user", user_message)
        self.memory.append("assistant", answer)
        return answer
