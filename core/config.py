from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    model_id: str = os.getenv("RON_MODEL_ID", "Qwen/Qwen3-1.7B")
    max_new_tokens: int = int(os.getenv("RON_MAX_NEW_TOKENS", "512"))
    temperature: float = float(os.getenv("RON_TEMPERATURE", "0.7"))
    top_p: float = float(os.getenv("RON_TOP_P", "0.9"))
    memory_path: str = os.getenv("RON_MEMORY_PATH", "data/memory.json")


settings = Settings()
