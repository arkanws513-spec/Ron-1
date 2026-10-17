"""Ron-1 inference API powered by pretrained SmolLM weights."""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.environ.get("RON1_MODEL_ID", "HuggingFaceTB/SmolLM3-3B")
MODEL_NAME = "Ron-1 SmolLM"
ALLOWED_ORIGIN = os.environ.get("RON1_ALLOWED_ORIGIN", "https://arkanws513-spec.github.io")
MAX_BODY_BYTES = 64 * 1024
MAX_INPUT_TOKENS = max(128, int(os.environ.get("RON1_MAX_INPUT_TOKENS", "2048")))
MAX_NEW_TOKENS = max(16, min(int(os.environ.get("RON1_MAX_NEW_TOKENS", "256")), 512))
torch.set_num_threads(max(1, int(os.environ.get("RON1_TORCH_THREADS", "2"))))
SYSTEM_PROMPT = os.environ.get(
    "RON1_SYSTEM_PROMPT",
    "أنت رون، مساعد ذكي ومفيد. أجب بلغة المستخدم، وكن واضحًا ودقيقًا. "
    "إذا لم تعرف الإجابة فقل ذلك بوضوح، ولا تخترع معلومات أو تدّع تنفيذ أفعال لم تنفذها.",
)

_model = None
_tokenizer = None
_load_error = ""
_load_lock = threading.Lock()
_generation_lock = threading.Lock()


def get_model():
    """Load pretrained weights once; never report random or missing weights as ready."""
    global _model, _tokenizer, _load_error
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer
    with _load_lock:
        if _model is not None and _tokenizer is not None:
            return _model, _tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=False,
            )
            model.eval()
            _tokenizer, _model = tokenizer, model
            _load_error = ""
        except Exception as exc:
            _load_error = (
                "تعذر تحميل أوزان النموذج المسبق التدريب. تحقق من الذاكرة "
                "واتصال الاستضافة ومستودع النموذج. " + str(exc)[:500]
            )
            raise RuntimeError(_load_error) from exc
    return _model, _tokenizer


def sanitize_history(history, message):
    """Bound and validate chat history before formatting it for the model."""
    if not isinstance(history, list):
        return []
    safe = []
    for item in history[-12:]:
        if not isinstance(item, dict):
            continue
        role, content = item.get("role"), item.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        content = content.strip()
        if content:
            safe.append({"role": role, "content": content[:4000]})
    if not safe or safe[-1]["role"] != "user" or safe[-1]["content"] != message:
        safe.append({"role": "user", "content": message[:4000]})
    return safe


def build_inputs(tokenizer, message, history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(sanitize_history(history, message))
    if getattr(tokenizer, "chat_template", None):
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if isinstance(encoded, dict):
            input_ids = encoded["input_ids"]
            attention_mask = encoded.get("attention_mask")
        else:
            input_ids, attention_mask = encoded, None
    else:
        lines = [SYSTEM_PROMPT]
        for item in messages[1:]:
            label = "المستخدم" if item["role"] == "user" else "رون"
            lines.append(f"{label}: {item['content']}")
        lines.append("رون:")
        encoded = tokenizer(
            "\n".join(lines),
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")

    if input_ids.shape[-1] > MAX_INPUT_TOKENS:
        input_ids = input_ids[:, -MAX_INPUT_TOKENS:]
        if attention_mask is not None:
            attention_mask = attention_mask[:, -MAX_INPUT_TOKENS:]
    return input_ids, attention_mask


def health_payload():
    try:
        model, tokenizer = get_model()
        return {
            "model": MODEL_NAME,
            "base_model": MODEL_ID,
            "ready": model is not None and tokenizer is not None,
            "trained": True,
            "weights_source": "pretrained Hugging Face checkpoint",
            "message": "تم تحميل أوزان SmolLM المحددة بنجاح.",
        }
    except Exception:
        return {
            "model": MODEL_NAME,
            "base_model": MODEL_ID,
            "ready": False,
            "trained": False,
            "weights_source": "pretrained Hugging Face checkpoint",
            "message": _load_error or "لم يتم تحميل النموذج بعد.",
        }


class Handler(BaseHTTPRequestHandler):
    server_version = "Ron1SmolLM/1.0"

    def _send(self, status, payload):
        body = b"" if status == 204 else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", ALLOWED_ORIGIN)
        self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Accept")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(204, {})

    def do_GET(self):
        if self.path.rstrip("/") in ("/health", "/api/health"):
            payload = health_payload()
            self._send(200 if payload["ready"] else 503, payload)
        else:
            self._send(404, {"model": MODEL_NAME, "error": "المسار غير موجود."})

    def do_POST(self):
        if self.path.rstrip("/") != "/api/chat":
            self._send(404, {"model": MODEL_NAME, "error": "المسار غير موجود."})
            return
        try:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except (TypeError, ValueError):
                length = 0
            if length <= 0 or length > MAX_BODY_BYTES:
                self._send(413, {"model": MODEL_NAME, "error": "حجم الطلب غير صالح أو كبير."})
                return
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(data, dict):
                self._send(400, {"model": MODEL_NAME, "error": "صيغة الطلب غير صالحة."})
                return
            message = data.get("message", "")
            if not isinstance(message, str) or not message.strip():
                self._send(400, {"model": MODEL_NAME, "error": "اكتب رسالة أولًا."})
                return
            message = message.strip()[:4000]
            model, tokenizer = get_model()
            input_ids, attention_mask = build_inputs(tokenizer, message, data.get("history", []))
            input_ids = input_ids.to(next(model.parameters()).device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(input_ids.device)
            generation_args = {
                "input_ids": input_ids,
                "max_new_tokens": MAX_NEW_TOKENS,
                "do_sample": True,
                "temperature": max(0.1, min(float(os.environ.get("RON1_TEMPERATURE", "0.7")), 1.5)),
                "top_p": max(0.1, min(float(os.environ.get("RON1_TOP_P", "0.9")), 1.0)),
                "repetition_penalty": 1.08,
                "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if attention_mask is not None:
                generation_args["attention_mask"] = attention_mask
            with _generation_lock, torch.inference_mode():
                output = model.generate(**generation_args)
            generated = output[0, input_ids.shape[-1]:]
            reply = tokenizer.decode(generated, skip_special_tokens=True).strip()
            if not reply:
                reply = "لم ينتج النموذج ردًا واضحًا هذه المرة. أعد صياغة السؤال من فضلك."
            self._send(200, {
                "model": MODEL_NAME,
                "base_model": MODEL_ID,
                "reply": reply,
                "weights_source": "pretrained Hugging Face checkpoint",
            })
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._send(400, {"model": MODEL_NAME, "error": "تعذر قراءة الطلب: " + str(exc)})
        except RuntimeError as exc:
            self._send(503, {"model": MODEL_NAME, "error": str(exc)[:700]})
        except Exception:
            self._send(500, {"model": MODEL_NAME, "error": "حدث خطأ داخلي أثناء توليد الرد."})

    def log_message(self, fmt, *args):
        print("%s - %s" % (self.address_string(), fmt % args), flush=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"Starting Ron-1 SmolLM API on {host}:{port}", flush=True)
    print(f"Pretrained model: {MODEL_ID}", flush=True)
    ThreadingHTTPServer((host, port), Handler).serve_forever()
