# Ron-1

Ron-1 is a chat interface and inference service using **pretrained SmolLM weights**. The project no longer requires its tiny randomly initialized native Transformer to answer conversations.

## Architecture

- **Web interface:** GitHub Pages at https://arkanws513-spec.github.io/Ron-1/
- **Inference API:** Python service in `core/server.py`.
- **Default pretrained model:** [HuggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B), a 3B-parameter multilingual conversational model whose model card lists Arabic among supported languages.
- **Model loading:** `transformers` downloads and caches the selected model on the API host. The weights are not committed to this Git repository or included in the browser bundle.
- **No hidden fallback:** if the selected checkpoint cannot be loaded, health reports not-ready and chat returns an error instead of generating random text.

## Run the API

Use Python 3.11 or newer:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r core/requirements.txt
python core/server.py
```

The first launch downloads model files from Hugging Face, so it needs outbound internet and sufficient disk space/RAM. The default 3B model is substantially larger than the old experimental core; plan for at least 8 GB RAM as a starting point, with more headroom preferred. CPU inference may be slow. A GPU is recommended for a responsive public service.

Check `GET /health` and wait for `ready: true`. Send conversations to `POST /api/chat`:

```json
{
  "message": "ما عاصمة مصر؟",
  "history": [
    {"role": "user", "content": "أجبني بالعربية"},
    {"role": "assistant", "content": "حاضر، سأجيب بالعربية."}
  ]
}
```

## Configuration

- `PORT`: service port; defaults to 8080.
- `HOST`: bind address; defaults to `0.0.0.0`.
- `RON1_MODEL_ID`: Hugging Face model ID; defaults to `HuggingFaceTB/SmolLM3-3B`.
- `RON1_ALLOWED_ORIGIN`: exact allowed web origin; defaults to `https://arkanws513-spec.github.io`.
- `RON1_MAX_INPUT_TOKENS`: maximum prompt tokens; defaults to 2048.
- `RON1_MAX_NEW_TOKENS`: generation limit; defaults to 256, capped at 512.
- `RON1_TEMPERATURE` and `RON1_TOP_P`: generation sampling controls.
- `RON1_TORCH_THREADS`: CPU thread count; defaults to 2.
- `RON1_SYSTEM_PROMPT`: optional system instruction.

The model uses its tokenizer's chat template when available and carries bounded recent conversation history. Generation is serialized to reduce RAM spikes on small CPU hosts.

## Deploy

Build and run the included Docker image as a separate API service. Make sure the host has enough RAM and outbound access to download the model. Mount a persistent volume at `/data/huggingface` to avoid downloading model files again after every restart, if the host supports persistent storage. Set `RON1_ALLOWED_ORIGIN` to the exact deployed interface origin if you use a different URL.

GitHub Pages only publishes `index.html` and `sw.js`; it cannot run PyTorch or host the inference service. Enter the API base URL in the page and press **اتصال**.

## About the original native core

`core/ron1_core.py`, `core/initialize.py`, and `core/train.py` remain available as an experimental from-scratch research path. Its small random initialization is **not** the conversational model used by the API. Training that experimental model does not fine-tune SmolLM.

## Limitations

SmolLM is a compact model, not GPT-3-class intelligence. It can make mistakes and Arabic quality varies by task. A reliable “professional” assistant still needs a fixed Arabic/English evaluation set, tested deployment hardware, and task-specific fine-tuning if evaluation shows it is needed. The API integration alone is not proof of quality or a claim that deployment is already live.
