"""Generate text with Ron-1's native checkpoint."""
import argparse
from pathlib import Path

import torch

from ron1_core import ByteTokenizer, Ron1Core, load_config


def load_trained_model(weights_path):
    """Load only a checkpoint explicitly marked as trained by Ron-1."""
    path = Path(weights_path)
    if not path.is_file():
        raise SystemExit(
            f"Weights missing: {path}. Train Ron-1 before generating text."
        )

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise SystemExit(f"Could not read checkpoint {path}: {exc}") from exc

    if not isinstance(checkpoint, dict):
        raise SystemExit("Invalid checkpoint: expected a Ron-1 checkpoint object.")
    if checkpoint.get("format") != "ron1-native-state-dict-v1":
        raise SystemExit("Unsupported checkpoint format; refusing to load unknown weights.")
    if checkpoint.get("trained") is not True or int(checkpoint.get("step", 0)) < 1:
        raise SystemExit(
            "This checkpoint contains initialized or untrained weights. "
            "Run core/train.py first; Ron-1 will not present random-weight output as a response."
        )
    if not isinstance(checkpoint.get("state_dict"), dict) or not checkpoint["state_dict"]:
        raise SystemExit("Invalid checkpoint: model state is missing.")

    config = checkpoint.get("config") or load_config()
    model = Ron1Core(config)
    try:
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    except Exception as exc:
        raise SystemExit(f"Checkpoint weights do not match the configured model: {exc}") from exc
    model.eval()
    return model, ByteTokenizer(), checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="core/weights/ron1-native.pt")
    parser.add_argument("--prompt", default="أهلاً، أنا رون.")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    if args.max_new_tokens < 1:
        raise SystemExit("--max-new-tokens must be at least 1.")
    if args.temperature < 0:
        raise SystemExit("--temperature cannot be negative.")

    model, tokenizer, _ = load_trained_model(args.weights)
    input_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print(tokenizer.decode(output_ids[0].tolist()[input_ids.shape[1]:]))


if __name__ == "__main__":
    main()
