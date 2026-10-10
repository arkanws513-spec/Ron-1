#!/usr/bin/env python3
"""Merge a trained PEFT LoRA adapter into a Hugging Face causal LM checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    if not args.adapter_dir.is_dir():
        raise FileNotFoundError(f"Adapter directory not found: {args.adapter_dir}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty or absent: {args.output_dir}")

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype)
    adapted = PeftModel.from_pretrained(base, str(args.adapter_dir))
    merged = adapted.merge_and_unload()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(str(args.adapter_dir), use_fast=True)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Merged checkpoint saved to {args.output_dir}")
    print("This is not yet a browser-ready ONNX/Q4F16 bundle; export, quantize, and test it separately.")


if __name__ == "__main__":
    main()
