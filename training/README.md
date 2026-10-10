# Ron-1 LoRA training workflow

This folder adds a real, reproducible fine-tuning path for the SmolLM2 base model. It is deliberately separate from browser inference: the deployed Q4F16 ONNX file is quantized for inference and is not the checkpoint to train directly.

## Environment

Use Python 3.10+ and install the training dependencies in an isolated environment:

\`\`\`bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install -r requirements-training.txt
\`\`\`

Install a PyTorch build appropriate for the machine if the default package is not suitable for its CPU/CUDA setup. Training can run on CPU but will be substantially slower; a CUDA GPU is recommended.

## Dataset format

Create a UTF-8 JSON Lines file, one training example per line. Supported forms:

\`\`\`json
{"prompt":"اشرح دورة الماء باختصار.","response":"تتبخر المياه ثم تتكاثف في السحب وتهطل من جديد."}
{"messages":[{"role":"user","content":"ما عاصمة مصر؟"},{"role":"assistant","content":"القاهرة."}]}
{"text":"A complete already-formatted training example can also be supplied here."}
\`\`\`

Use only data you have permission to use. Remove private, sensitive, or identifying information. Keep examples accurate and representative of the behavior you want Ron to learn. The script validates JSONL and message roles before training.

## Fine-tune

\`\`\`bash
python training/train_lora.py \\
  --base-model HuggingFaceTB/SmolLM2-135M-Instruct \\
  --train-file ./data/ron-training.jsonl \\
  --output-dir ./training-output/ron1-smollm2-lora \\
  --epochs 3 --batch-size 2 --gradient-accumulation 8 \\
  --max-length 1024 --lora-r 8
\`\`\`

For a quick data/pipeline smoke test, use \`--limit 2 --epochs 1\`; this still loads the base model and is not a substitute for quality evaluation. Training downloads the base checkpoint to the training machine if it is not already cached.

## Merge adapter

\`\`\`bash
python training/merge_lora.py \\
  --base-model HuggingFaceTB/SmolLM2-135M-Instruct \\
  --adapter-dir ./training-output/ron1-smollm2-lora \\
  --output-dir ./training-output/ron1-merged
\`\`\`

## Important deployment boundary

A LoRA adapter or merged Hugging Face checkpoint is **not** automatically a new browser model. To replace the deployed Ron-1 weights, the merged checkpoint must be exported to a compatible ONNX graph, quantized to the exact runtime variant, checked against the tokenizer/config/KV-cache tensor types, hashed, and run through browser inference tests before publication. The current Pages workflow intentionally continues serving the known SmolLM2 Q4F16 artifact; it does not silently replace production weights with an unvalidated training result.

No trained weights were generated as part of adding this pipeline: a real dataset and training compute are required. CI validates Python syntax and repository contracts, not model quality.
