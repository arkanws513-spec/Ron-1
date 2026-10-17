import json
import os
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler

MODEL_ID = os.environ.get("RON1_MODEL_ID", "HuggingFaceTB/SmolLM3-3B")
ORIGIN = os.environ.get("RON1_ALLOWED_ORIGIN", "https://arkanws513-spec.github.io")
ROUTER = "https://router.huggingface.co/v1/chat/completions"
SYSTEM_PROMPT = os.environ.get(
    "RON1_SYSTEM_PROMPT",
    "أنت رون، مساعد ذكي ومفيد. أجب بلغة المستخدم بوضوح ودقة. إذا لم تعرف فقل ذلك ولا تخترع معلومات."
)

class handler(BaseHTTPRequestHandler):
    def send_json(self, status, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", ORIGIN)
        self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Accept")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_json(200, {"ok": True})

    def do_POST(self):
        token = os.environ.get("HF_TOKEN", "").strip()
        if not token:
            self.send_json(503, {"model": "Ron-1 SmolLM", "error": "الخدمة غير مهيأة: أضف HF_TOKEN إلى إعدادات Vercel."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0 or length > 65536:
                raise ValueError("حجم الطلب أكبر من الحد المسموح.")
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError("صيغة الطلب غير صالحة.")
            message = data.get("message")
            if not isinstance(message, str) or not message.strip():
                raise ValueError("اكتب رسالة أولًا.")
            message = message.strip()
            if len(message) > 4000:
                self.send_json(413, {"model": "Ron-1 SmolLM", "error": "الرسالة أطول من الحد المسموح (4000 حرف)."})
                return
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            history = data.get("history", [])
            if isinstance(history, list):
                for item in history[-12:]:
                    if not isinstance(item, dict):
                        continue
                    role, content = item.get("role"), item.get("content")
                    if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                        messages.append({"role": role, "content": content.strip()[:4000]})
            if not any(m["role"] == "user" and m["content"] == message for m in messages[1:]):
                messages.append({"role": "user", "content": message})
            try:
                max_tokens = max(16, min(int(os.environ.get("RON1_MAX_NEW_TOKENS", "256")), 512))
            except ValueError:
                max_tokens = 256
            try:
                temperature = max(0.0, min(float(os.environ.get("RON1_TEMPERATURE", "0.7")), 2.0))
            except ValueError:
                temperature = 0.7
            payload = {"model": MODEL_ID, "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages[1:][-12:],
                       "max_tokens": max_tokens, "temperature": temperature, "stream": False}
            req = urllib.request.Request(ROUTER, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Authorization": "Bearer " + token, "Content-Type": "application/json", "Accept": "application/json"}, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=52) as response:
                    result = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read(3000).decode("utf-8", errors="replace")
                try:
                    obj = json.loads(detail)
                    detail = str(obj.get("error") or obj.get("message") or detail)
                except json.JSONDecodeError:
                    pass
                if exc.code in (401, 403):
                    error = "رمز HF_TOKEN غير صالح أو لا يملك صلاحية Inference Providers."
                elif exc.code == 402:
                    error = "مزود الاستدلال يطلب رصيدًا أو إعداد فوترة لهذا النموذج."
                elif exc.code == 429:
                    error = "تم تجاوز حد الطلبات؛ انتظر قليلًا ثم أعد المحاولة."
                elif exc.code == 404:
                    error = "النموذج غير متاح لدى مزودي الاستدلال حاليًا. راجع RON1_MODEL_ID."
                else:
                    error = "تعذر توليد الرد: " + detail[:400]
                self.send_json(502, {"model": "Ron-1 SmolLM", "error": error})
                return
            except (urllib.error.URLError, TimeoutError):
                self.send_json(504, {"model": "Ron-1 SmolLM", "error": "انتهت مهلة الاتصال بمزود الاستدلال. حاول مجددًا."})
                return
            choices = result.get("choices", []) if isinstance(result, dict) else []
            reply = ""
            if choices and isinstance(choices[0], dict):
                msg = choices[0].get("message", {})
                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    reply = msg["content"].strip()
            if not reply:
                self.send_json(502, {"model": "Ron-1 SmolLM", "error": "وصل رد غير متوقع من مزود الاستدلال."})
                return
            self.send_json(200, {"model": "Ron-1 SmolLM", "base_model": result.get("model", MODEL_ID), "reply": reply, "usage": result.get("usage")})
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            self.send_json(400, {"model": "Ron-1 SmolLM", "error": str(exc) or "أرسل طلبًا صالحًا."})
        except Exception:
            self.send_json(500, {"model": "Ron-1 SmolLM", "error": "حدث خطأ داخلي أثناء تجهيز الطلب."})

    def log_message(self, fmt, *args):
        return
