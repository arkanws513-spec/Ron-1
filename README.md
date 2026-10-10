# Ron-1

Ron-1 is being migrated to its own project-owned native Transformer core. The browser interface is a separate static application; the model weights and inference runtime are not bundled into the page.

## Current architecture

- **Interface:** GitHub Pages, served from \`index.html\`.
- **Native model architecture:** \`core/ron1_core.py\`, a decoder-only Transformer with a UTF-8 byte tokenizer.
- **Native inference API:** \`core/server.py\`, exposing \`GET /health\` and \`POST /api/chat\`.
- **Native checkpoint:** supplied separately at \`core/weights/ron1-native.pt\` or via \`RON1_WEIGHTS_PATH\`.
- **No SmolLM2 fallback:** the interface does not load SmolLM2, ONNX weights, Transformers.js, or a third-party model endpoint.

The Pages workflow intentionally publishes only \`index.html\` and \`sw.js\`. This keeps model weights out of the interface bundle. Enter the deployed native API base URL in the interface and select **Connect**.

## Important model status

The native architecture and initializer are project-owned, but the initializer creates random weights, not a useful trained language model. The API deliberately refuses to generate responses until it finds a checkpoint with \`trained: true\` and a positive training step. This is intentional: Ron-1 will not silently fall back to SmolLM2 or another provider.

## Initialize and train the native core

Install the dependency:

\`\`\`bash
python -m pip install -r core/requirements.txt
python core/initialize.py
\`\`\`

Train using UTF-8 text files that you have permission to use:

\`\`\`bash
python core/train.py --data data/ron1 --steps 1000 --batch-size 16
\`\`\`

The small v0.1 architecture is a development starting point, not yet a production-grade LLM. A curated training corpus and substantial training/evaluation are required before useful conversation should be expected.

## Run the native API locally

\`\`\`bash
python core/server.py
\`\`\`

By default it listens on port 8080. Set \`RON1_WEIGHTS_PATH\` to the trained checkpoint location and \`RON1_ALLOWED_ORIGIN\` to the exact interface origin for deployment. The default CORS origin is \`*\` for simple local testing; production deployments should restrict it.

A Dockerfile is included for a separate API service. The trained checkpoint is intentionally not committed or baked into the image; mount/provide it separately and set \`RON1_WEIGHTS_PATH\`.

## Deployment

- Static interface: GitHub Pages.
- Native inference: a separate Python service using the repository's Dockerfile.
- The API service must have trained native weights available. The public Pages interface cannot run PyTorch directly.

See [DEPLOYMENT.md](DEPLOYMENT.md) for setup details.
