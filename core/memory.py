import json
import os
from pathlib import Path
from typing import Any


class Memory:
    def __init__(self, path: str, max_messages: int = 20):
        self.path = Path(path)
        self.max_messages = max_messages
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> list[dict[str, str]]:
        if not self.path.exists():
            return []
        try:
            data: Any = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                return []
            return [
                item for item in data
                if isinstance(item, dict)
                and item.get("role") in {"user", "assistant"}
                and isinstance(item.get("content"), str)
            ][-self.max_messages:]
        except (OSError, json.JSONDecodeError):
            return []

    def save(self, messages: list[dict[str, str]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        trimmed = messages[-self.max_messages:]
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        payload = json.dumps(trimmed, ensure_ascii=False, indent=2)
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.path)
