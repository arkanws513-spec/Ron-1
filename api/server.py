import os
import secrets

from fastapi import FastAPI, Header, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from core.ron import Ron

app = FastAPI(title="Ron-1", version="1.0.0")
ron = Ron()


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=20_000)
    # Supplied by Ron Cloud only after it authenticates the account.
    user_id: str = Field(min_length=1, max_length=256)
    conversation_id: str = Field(default="default", min_length=1, max_length=128)


@app.get("/health")
def health() -> dict[str, str]:
    """Lightweight liveness check; does not load the model."""
    return {"status": "ok", "assistant": "Ron-1"}


@app.post("/chat")
def chat(
    request: ChatRequest,
    x_ron_api_key: str | None = Header(default=None, alias="X-Ron-API-Key"),
) -> dict[str, str]:
    configured_key = os.getenv("RON_API_KEY", "")
    if not configured_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat service is not configured.",
        )
    if not x_ron_api_key or not secrets.compare_digest(x_ron_api_key, configured_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API credentials.",
        )

    message = request.message.strip()
    if not message:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Message must not be blank.",
        )

    return {
        "response": ron.chat(
            message,
            user_id=request.user_id,
            conversation_id=request.conversation_id,
        )
    }
