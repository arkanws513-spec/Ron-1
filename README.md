---
title: Ron-1
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
---

# Ron-1

ذكاء اصطناعي للمستقبل.

Ron-1 uses Qwen3-1.7B as its base language model and adds a custom Ron Core for conversation, memory, reasoning, and task orchestration.

## Model

The official Qwen3-1.7B model files are stored in the Ron-1 GitHub Release `qwen3-1.7b-weights-v1`.

The largest safetensors file is stored as two release assets and reconstructed before loading. Ron-1 downloads these assets into its local model directory on first startup, then loads the reconstructed model with `local_files_only=True`.

This means Ron's inference path is local model execution, not a hosted inference API.

## Run

```bash
pip install -r requirements.txt
python -m model.download_model
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

For the Hugging Face Space, `app.py` provides the chat UI directly.

Qwen3-1.7B is distributed by Qwen under Apache-2.0. Ron-1 is independent and is not an official Qwen or Alibaba product.
