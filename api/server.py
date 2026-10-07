from fastapi import FastAPI
from pydantic import BaseModel

from core.ron import Ron

app = FastAPI(title="Ron-1")
ron = Ron()


class ChatRequest(BaseModel):
    message: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "assistant": "Ron-1"}


@app.post("/chat")
def chat(request: ChatRequest) -> dict[str, str]:
    return {"response": ron.chat(request.message)}
