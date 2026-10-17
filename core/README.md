# Ron-1 Experimental Native Core

This directory preserves Ron-1's original from-scratch decoder-only Transformer as a research experiment. It is **not** the default conversation engine.

## Production conversation path

The web interface now connects to `core/server.py`, which loads pretrained SmolLM weights through Hugging Face Transformers. The default model is `HuggingFaceTB/SmolLM3-3B`. See the repository root README and `DEPLOYMENT.md` for the supported runtime.

## Experimental native path

The native initializer creates deterministic random starting weights, **not a trained conversational model**. Random weights do not contain language knowledge.

Install dependencies and initialize the experimental checkpoint:

```bash
python -m pip install -r core/requirements.txt
python core/initialize.py
```

Train this separate experimental architecture on UTF-8 text files you have permission to use:

```bash
python core/train.py --data data/ron1 --steps 1000 --batch-size 16
```

Training uses next-byte prediction with a small byte tokenizer. Training this native architecture does **not** fine-tune SmolLM and does not change the weights used by the production API.

## Architecture v0.1

- Decoder-only Transformer with causal self-attention and feed-forward blocks.
- UTF-8 byte tokenizer (259 IDs).
- Tied input/output embeddings.
- Next-token cross-entropy objective and autoregressive generation.
- Four layers, dimension 128 and a 256-token context: a small development prototype, not a production-grade LLM.

Do not describe the native model as trained or conversationally capable unless a separate evaluation demonstrates that result.
