# Ron-1 Native Core

This directory introduces a new decoder-only Transformer architecture owned by the Ron-1 project. It does not load SmolLM2, ONNX weights, a pretrained checkpoint, or a remote inference service. The trainable parameters in `Ron1Core` are Ron-1's own model weights.

## Honest status

The initializer creates deterministic random starting weights, **not a trained conversational model**. Random weights do not contain language knowledge. Ron-1 must be trained on a curated, appropriately licensed text corpus before it can answer usefully. No training is claimed just because the architecture or checkpoint exists.

## Initialize weights

From the repository root:

```bash
python -m pip install -r core/requirements.txt
python core/initialize.py
```

This creates `core/weights/ron1-native.pt` with Ron-1's state dictionary, architecture configuration, step counter and training flag.

## Train

Provide UTF-8 text files you have permission to use:

```bash
python core/train.py --data data/ron1 --steps 1000 --batch-size 16
```

Continue from an existing checkpoint with `--resume`. Training uses next-token prediction over UTF-8 bytes. Curate data for quality, duplicates, privacy and safety before training.

## Generate locally

```bash
python core/generate.py --prompt "اكتب تعريفًا بنفسك"
```

## Architecture v0.1

- Decoder-only Transformer, causal self-attention and feed-forward blocks.
- UTF-8 byte tokenizer (259 IDs) that supports Arabic without an external tokenizer.
- Tied input/output embeddings.
- Next-token cross-entropy objective and autoregressive generation.
- Small development configuration; not yet a production-quality LLM.

This is the first native-core implementation. The existing public website still uses the previous SmolLM2 runtime and has not been migrated to this core.
