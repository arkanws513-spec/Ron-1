import json
import os
import time
import urllib.error
import urllib.request
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="Ron-1 Chat API", version="1.1.0")

ALLOWED_ORIGINS = {
    origin.strip()
    for origin in os.getenv(
        "RON_ALLOWED_ORIGINS",
        "https://arkanws513-spec.github.io",
    ).split(",")
    if origin.strip()
}
app.add_middleware(
    CORSMiddleware,
    allow_origins=sorted(ALLOWED_ORIGINS),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

_requests = defaultdict(deque)
RATE_LIMIT = int(os.getenv("RON_RATE_LIMIT_PER_MINUTE", "12"))


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    conversation_id: str = Field(default="default", min_length=1, max_length=128)


def check_rate_limit(request: Request) -> None:
    now = time.time()
    client_ip = request.client.host if request.client else "unknown"
    bucket = _requests[client_ip]
    while bucket and now - bucket[0] > 60:
        bucket.popleft()
    if len(bucket) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="تم تجاوز حد الرسائل مؤقتًا. حاول بعد دقيقة.")
    bucket.append(now)


@app.get("/")
def root():
    return {"service": "Ron-1", "status": "ready", "docs": "/docs"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "assistant": "Ron-1",
        "provider_configured": bool(os.getenv("HF_TOKEN")),
        "model": os.getenv("HF_MODEL", "Qwen/Qwen3-4B-Instruct-2507"),
    }


@app.post("/chat")
def chat(body: ChatRequest, request: Request):
    check_rate_limit(request)
    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        raise HTTPException(
            status_code=503,
            detail="خدمة النموذج لم تُضبط بعد. أضف HF_TOKEN في متغيرات Railway.",
        )

    model = os.getenv("HF_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "أنت Ron-1، مساعد عربي مفيد وواضح. أجب بلغة المستخدم، "
                    "ولا تدّعِ أنك نفذت إجراءات لم تنفذها."
                ),
            },
            {"role": "user", "content": body.message.strip()},
        ],
        "max_tokens": int(os.getenv("RON_MAX_TOKENS", "700")),
        "stream": False,
    }
    req = urllib.request.Request(
        "https://router.huggingface.co/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as response:
            result = json.loads(response.read().decode("utf-8"))
        answer = result["choices"][0]["message"]["content"]
        if isinstance(answer, list):
            answer = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in answer
            )
        return {"response": str(answer), "model": model}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        if exc.code in (401, 403):
            raise HTTPException(
                status_code=502,
                detail="رمز Hugging Face غير صالح أو لا يملك صلاحية استخدام مزود الاستدلال.",
            ) from exc
        if exc.code == 429:
            raise HTTPException(
                status_code=503,
                detail="مزود النموذج بلغ حد الاستخدام المجاني مؤقتًا. حاول لاحقًا.",
            ) from exc
        raise HTTPException(
            status_code=502,
            detail=f"تعذر على مزود النموذج إكمال الطلب (HTTP {exc.code}).",
        ) from exc
    except (urllib.error.URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=502,
            detail="لم يتمكن رون من الاتصال بمزود النموذج. تحقق من إعداداته وحاول مجددًا.",
        ) from exc
