import os
from http.server import BaseHTTPRequestHandler
import json

MODEL_ID = os.environ.get("RON1_MODEL_ID", "HuggingFaceTB/SmolLM3-3B")
ORIGIN = os.environ.get("RON1_ALLOWED_ORIGIN", "https://arkanws513-spec.github.io")

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

    def do_GET(self):
        configured = bool(os.environ.get("HF_TOKEN", "").strip())
        self.send_json(200 if configured else 503, {
            "model": "Ron-1 SmolLM",
            "base_model": MODEL_ID,
            "ready": configured,
            "trained": configured,
            "weights_source": "Hugging Face Inference Providers",
            "message": ("خدمة الاستدلال مهيأة؛ سيُختبر التوليد عند إرسال أول رسالة."
                        if configured else
                        "أضف HF_TOKEN بصلاحية Inference Providers إلى متغيرات بيئة Vercel.")
        })

    def log_message(self, fmt, *args):
        return
