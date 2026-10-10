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

## Run locally (Gradio interface)

Requires Python 3.10+ and enough RAM/storage for the model weights.

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

The Gradio interface will print its local URL in the terminal. On first startup, downloading and loading the model can take time and requires internet access and disk space.

## Run the API with Docker

The API is intended to run on a server that can store and load the model. It is not a static website.

```bash
docker build -t ron-1 .
docker run --rm -p 8000:8000 \
  -e RON_API_KEY="replace-with-a-long-random-secret" \
  -v ron-1-data:/data \
  ron-1
```

Check liveness at `GET /health`. Chat requests use `POST /chat` with JSON containing `message`, `user_id`, and optionally `conversation_id`, plus the `X-Ron-API-Key` header. Do not expose the API publicly without setting a strong `RON_API_KEY` and configuring HTTPS, access controls, and resource limits.

For private release assets, configure a GitHub token with read-only access to the required release assets in the deployment platform's secret manager. Never commit tokens, API keys, or model weights into the repository.

## GitHub publishing and deployment

This repository is hosted on GitHub. GitHub Pages only serves static files; it cannot run the Python inference service or host/load the model as a live backend. Use GitHub Actions for automated checks and a compatible Python/container host for a live API. The `.github/workflows/ci.yml` workflow compiles Python modules, runs tests, builds the Docker image, and performs basic container health/auth checks on pushes and pull requests.

Before a public production launch:
- Review the GitHub Actions run and resolve any failing checks.
- Store deployment credentials as repository/environment secrets, never as committed files.
- Confirm the selected host has sufficient memory, disk space, and CPU/GPU resources for Qwen3-1.7B.
- Add a project-code license only after choosing the intended license for Ron-1's own code; the base model's license does not automatically license all project code.

## Project structure

- `app.py` — Gradio chat interface.
- `api/` — FastAPI server.
- `core/` — Ron conversation, reasoning, and memory logic.
- `model/` — model backend and weight-loading utilities.
- `Dockerfile` — container image for the API.
- `requirements.txt` / `requirements-api.txt` — interface and API dependencies.
- `.github/workflows/ci.yml` — automated validation.

## Security

Please do not publish secrets, access tokens, personal conversation logs, or model files in issues or pull requests. Report suspected vulnerabilities privately to the repository owner until a dedicated security contact is configured.
