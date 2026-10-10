import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="Ron-1 Chat API", version="1.2.0")

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


def local_reply(message: str):
    """Reliable, zero-cost replies for basic conversation when inference is unavailable."""
    normalized = re.sub(r"[؟?!.,،؛:…ـ\s]+", " ", message.casefold()).strip()
    compact = normalized.replace(" ", "")
    greetings = {
        "مرحبا", "مرحباً", "اهلا", "أهلا", "اهلاوسهلا", "السلامعليكم",
        "صباحالخير", "مساءالخير", "hi", "hello", "hey", "goodmorning",
        "goodevening",
    }
    if compact in greetings or compact in {"السلامعليكمورحمةالله", "سلام"}:
        return "أهلًا بيك! أنا رون. أقدر أساعدك، لكن قدرات المحادثة المتقدمة متوقفة مؤقتًا لحين إصلاح اتصال نموذج الذكاء الاصطناعي."
    if compact in {"شكرا", "شكرًا", "متشكر", "متشكرة", "thanks", "thankyou"}:
        return "العفو! أنا هنا للمساعدة."
    if compact in {"ازيك", "إزيك", "عاملإيه", "كيفحالك", "howareyou"}:
        return "أنا جاهز للمساعدة. اتصال نموذج الذكاء الاصطناعي المتقدم يحتاج إصلاحًا، لكن استقبال الرسائل الأساسية يعمل."
    if compact in {"انت مين", "انتا مين", "من انت", "منأنت", "اسمكايه", "مااسمك", "whoareyou"}:
        return "أنا Ron-1، مشروع مساعد ذكاء اصطناعي. واجهة المحادثة والخادم موجودان، لكنني لا أملك حاليًا نموذجًا محليًا مستقلًا؛ التوليد المتقدم يعتمد على مزوّد خارجي."
    if compact in {"هل انت شغال", "انت شغال", "اختبار", "test", "ping"}:
        return "وصلتني رسالتك بنجاح. واجهة Ron-1 والخادم يستجيبان؛ ما زال اتصال نموذج الذكاء الاصطناعي الخارجي بحاجة إلى إصلاح."
    if compact in {"مع السلامة", "سلام", "باي", "bye", "goodbye"}:
        return "مع السلامة! أنا موجود لما ترجع."
    return None


@app.get("/")
def root():
    return {"service": "Ron-1", "status": "ready", "docs": "/docs"}


@app.get("/health")
def health():
    openai_key = bool(os.getenv("AI_API_KEY", "").strip())
    hf_key = bool(os.getenv("HF_TOKEN", "").strip())
    return {
        "status": "ok",
        "assistant": "Ron-1",
        "provider_configured": openai_key or hf_key,
        "provider": "openai-compatible" if openai_key else ("huggingface" if hf_key else "local-fallback"),
        "model": os.getenv("AI_MODEL") if openai_key else os.getenv("HF_MODEL", "Qwen/Qwen3-4B-Instruct-2507"),
        "local_fallback": True,
    }


@app.post("/chat")
def chat(body: ChatRequest, request: Request):
    check_rate_limit(request)
    message = body.message.strip()

    # Basic greetings and status checks should never fail because a paid/external
    # inference provider is unavailable. These are explicit local replies, not LLM output.
    fallback = local_reply(message)
    if fallback is not None:
        return {"response": fallback, "model": "ron-1-local-fallback", "mode": "local-fallback"}

    ai_key = os.getenv("AI_API_KEY", "").strip()
    ai_base = os.getenv("AI_BASE_URL", "").strip().rstrip("/")
    ai_model = os.getenv("AI_MODEL", "").strip()

    if ai_key and ai_base and ai_model:
        endpoint = ai_base if ai_base.endswith("/chat/completions") else ai_base + "/chat/completions"
        model = ai_model
        headers = {"Authorization": f"Bearer {ai_key}", "Content-Type": "application/json"}
    else:
        token = os.getenv("HF_TOKEN", "").strip()
        if not token:
            raise HTTPException(
                status_code=503,
                detail="الردود الأساسية تعمل، لكن نموذج الذكاء الاصطناعي غير مضبوط. أضف AI_API_KEY وAI_BASE_URL وAI_MODEL لمزوّد متوافق، أو أعد ضبط Hugging Face.",
            )
        model = os.getenv("HF_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
        endpoint = "https://router.huggingface.co/v1/chat/completions"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

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
            {"role": "user", "content": message},
        ],
        "max_tokens": int(os.getenv("RON_MAX_TOKENS", "700")),
        "stream": False,
    }
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
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
        return {"response": str(answer), "model": model, "mode": "inference"}
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            detail = "مفتاح المزوّد غير صالح أو لا يملك صلاحية استخدام النموذج."
        elif exc.code == 402:
            detail = "المزوّد رفض الطلب بسبب الرصيد أو الفوترة. لم يتم تفعيل أي دفع تلقائي من Ron-1."
        elif exc.code == 429:
            detail = "وصل المزوّد إلى حد الاستخدام مؤقتًا. جرّب لاحقًا أو اختر مزوّدًا آخر."
        else:
            detail = f"تعذر على مزوّد النموذج إكمال الطلب (HTTP {exc.code})."
        raise HTTPException(status_code=502, detail=detail) from exc
    except (urllib.error.URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=502,
            detail="تعذر الاتصال بمزوّد النموذج أو أن استجابته غير صالحة. تحقق من إعداداته.",
        ) from exc
