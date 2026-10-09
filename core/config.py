from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    # The local copy of the Qwen weights is the model Ron runs on.
    model_id: str = os.getenv(
        "RON_MODEL_ID", "models/Ron-1-Qwen3-1.7B"
    )
    upstream_model_id: str = os.getenv(
        "RON_UPSTREAM_MODEL_ID", "Qwen/Qwen3-1.7B"
    )
    max_new_tokens: int = int(os.getenv("RON_MAX_NEW_TOKENS", "512"))
    temperature: float = float(os.getenv("RON_TEMPERATURE", "0.7"))
    top_p: float = float(os.getenv("RON_TOP_P", "0.9"))
    # This is a directory root; each user/conversation gets its own file.
    memory_path: str = os.getenv("RON_MEMORY_PATH", "data/memory")


settings = Settings()
