---
title: Ron-1
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.49.1
python_version: "3.10"
app_file: app.py
startup_duration_timeout: 1h
short_description: المساعد المستقل Ron-1
---

# Ron-1

ذكاء اصطناعي للمستقبل.

Ron-1 uses Qwen3-1.7B as its base language model and adds a custom Ron Core for conversation, memory, reasoning, and task orchestration.

The official Qwen3-1.7B model files are stored in the Ron-1 GitHub Release `qwen3-1.7b-weights-v1`. The Space downloads those release assets on first startup, reconstructs the split safetensors file locally, and runs inference from the local weights.

Qwen3-1.7B is distributed by Qwen under Apache-2.0. Ron-1 is independent and is not an official Qwen or Alibaba product.
