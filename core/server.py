"""HTTP API for Ron-1's native, project-owned Transformer core."""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

CORE_DIR = Path(__file__).resolve().parent
ROOT_DIR = CORE_DIR.parent
sys.path.insert(0, str(CORE_DIR))
from ron1_core import ByteTokenizer, Ron1Core, load_config  # noqa: E402

WEIGHTS_PATH = Path(os.environ.get("RON1_WEIGHTS_PATH", str(CORE_DIR / "weights" / "ron1-native.pt")))
ALLOWED_ORIGIN = os.environ.get("RON1_ALLOWED_ORIGIN", "*")
MAX_BODY_BYTES = 64 * 1024
MAX_NEW_TOKENS = int(os.environ.get("RON1_MAX_NEW_TOKENS", "96"))
MODEL_NAME = "Ron-1 Native Core"
_cached_model = None
_cached_mtime = None
_cached_step = 0
_cached_error = "لم يتم العثور على ملف أوزان النواة الأصلية."
torch.set_num_threads(max(1, int(os.environ.get("RON1_TORCH_THREADS", "2"))))


def get_model():
    global _cached_model, _cached_mtime, _cached_step, _cached_error
    try:
        if not WEIGHTS_PATH.is_file():
            _cached_error = "ملف أوزان Ron-1 الأصلية غير موجود. درّب النواة أولًا."
            return None
        mtime = WEIGHTS_PATH.stat().st_mtime_ns
        if _cached_model is not None and _cached_mtime == mtime:
            return _cached_model
        checkpoint = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)
        if checkpoint.get("format") != "ron1-native-state-dict-v1":
            _cached_error = "صيغة ملف أوزان النواة غير مدعومة."
            return None
        if checkpoint.get("trained") is not True or int(checkpoint.get("step", 0)) < 1:
            _cached_error = "النواة الأصلية موجودة لكنها لم تُدرّب بعد؛ لن يتم استخدام نموذج بديل."
            return None
        config = checkpoint.get("config") or load_config()
        model = Ron1Core(config)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        _cached_model = model
        _cached_mtime = mtime
        _cached_step = int(checkpoint.get("step", 0))
        _cached_error = ""
        return _cached_model
    except Exception as exc:
        _cached_error = "تعذر تحميل أوزان Ron-1 الأصلية: " + str(exc)
        return None


def health_payload():
    model = get_model()
    return {
        "model": MODEL_NAME,
        "ready": model is not None,
        "trained": model is not None,
        "step": _cached_step,
        "message": "النواة الأصلية جاهزة." if model is not None else _cached_error,
    }


def make_prompt(message, history):
    parts = [
        "أنت رون، مساعد عربي مستقل. أجب عن رسالة المستخدم بوضوح.\n"
    ]
    safe_history = history if isinstance(history, list) else []
    for item in safe_history[-8:]:
        if not isinstance(item, dict):
            continue
        role, content = item.get("role"), item.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        content = content[:2000]
        parts.append(("المستخدم: " if role == "user" else "رون: ") + content + "\n")
    if not safe_history or not any(isinstance(item, dict) and item.get("role") == "user" and item.get("content") == message for item in safe_history[-8:]):
        parts.append("المستخدم: " + message[:2000] + "\n")
    parts.append("رون: ")
    return "".join(parts)


class Handler(BaseHTTPRequestHandler):
    server_version = "Ron1NativeCore/0.1"

    def _send(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", ALLOWED_ORIGIN)
        self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Accept")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(204, {})

    def do_GET(self):
        if self.path.rstrip("/") in ("/health", "/api/health"):
            self._send(200, health_payload())
        else:
            self._send(404, {"model": MODEL_NAME, "error": "المسار غير موجود."})

    def do_POST(self):
        if self.path.rstrip("/") != "/api/chat":
            self._send(404, {"model": MODEL_NAME, "error": "المسار غير موجود."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_BODY_BYTES:
                self._send(413, {"model": MODEL_NAME, "error": "حجم الطلب غير صالح أو كبير."})
                return
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            message = data.get("message", "")
            history = data.get("history", [])
            if not isinstance(message, str) or not message.strip():
                self._send(400, {"model": MODEL_NAME, "error": "اكتب رسالة أولًا."})
                return
            model = get_model()
            if model is None:
                self._send(503, {"model": MODEL_NAME, "error": _cached_error})
                return
            tokenizer = ByteTokenizer()
            prompt_ids = tokenizer.encode(make_prompt(message.strip(), history), add_bos=True)
            prompt_ids = prompt_ids[-model.config["max_seq_len"]:]
            input_ids = torch.tensor([prompt_ids], dtype=torch.long)
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=max(1, min(MAX_NEW_TOKENS, 160)),
                    temperature=0.7,
                    top_k=40,
                )
            reply = tokenizer.decode(output_ids[0].tolist()[len(prompt_ids):]).strip()
            if not reply:
                reply = "لم تنتج النواة نصًا واضحًا في هذه المحاولة."
            self._send(200, {"model": MODEL_NAME, "reply": reply, "step": _cached_step})
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._send(400, {"model": MODEL_NAME, "error": "تعذر قراءة الطلب: " + str(exc)})
        except Exception:
            self._send(500, {"model": MODEL_NAME, "error": "حدث خطأ داخلي أثناء تشغيل النواة الأصلية."})

    def log_message(self, fmt, *args):
        print("%s - %s" % (self.address_string(), fmt % args), flush=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    host = os.environ.get("HOST", "0.0.0.0")
    print("Starting Ron-1 Native Core API on %s:%s" % (host, port), flush=True)
    print("Weights path: %s" % WEIGHTS_PATH, flush=True)
    ThreadingHTTPServer((host, port), Handler).serve_forever()
