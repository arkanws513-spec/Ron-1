import json
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
            return data if isinstance(data, list) else []
        except (OSError, json.JSONDecodeError):
            return []

    def save(self, messages: list[dict[str, str]]) -> None:
        trimmed = messages[-self.max_messages:]
        self.path.write_text(
            json.dumps(trimmed, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def append(self, role: str, content: str) -> None:
        messages = self.load()
        messages.append({"role": role, "content": content})
        self.save(messages)
