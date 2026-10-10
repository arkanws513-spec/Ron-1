#!/usr/bin/env python3
"""LoRA fine-tuning entry point for Ron-1's SmolLM2 language-model base.

Training is an offline-development workflow, separate from browser inference.
Input JSONL supports {"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]},
{"prompt":"...", "response":"..."}, or {"text":"..."} records.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

LOG = logging.getLogger("ron1-training")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a causal language model with LoRA.")
    parser.add_argument("--train-file", required=True, type=Path, help="UTF-8 JSONL training data.")
    parser.add_argument("--base-model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--output-dir", type=Path, default=Path("training-output/ron1-smollm2-lora"))
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0, help="Optional small-record limit for smoke tests.")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    return parser.parse_args()


def read_records(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Training file does not exist: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_number} must contain a JSON object.")
            records.append(record)
            if limit > 0 and len(records) >= limit:
                break
    if not records:
        raise ValueError("Training file contains no usable records.")
    return records


def render_record(record: dict[str, Any], tokenizer: Any, index: int) -> str:
    if isinstance(record.get("text"), str) and record["text"].strip():
        return record["text"].strip()

    messages = record.get("messages")
    if messages is None and isinstance(record.get("prompt"), str) and isinstance(record.get("response"), str):
        messages = [
            {"role": "user", "content": record["prompt"]},
            {"role": "assistant", "content": record["response"]},
        ]
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"Record {index} needs 'text', 'messages', or both 'prompt' and 'response'.")

    cleaned = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant"}:
            raise ValueError(f"Record {index} has a message with an unsupported role.")
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError(f"Record {index} has an empty or invalid message content.")
        cleaned.append({"role": message["role"], "content": message["content"].strip()})
    if not any(message["role"] == "assistant" for message in cleaned):
        raise ValueError(f"Record {index} must include at least one assistant response.")
    return tokenizer.apply_chat_template(cleaned, tokenize=False, add_generation_prompt=False)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.epochs <= 0 or args.batch_size <= 0 or args.gradient_accumulation <= 0:
        raise ValueError("epochs, batch-size, and gradient-accumulation must be positive.")
    if args.max_length < 16 or args.lora_r <= 0 or args.lora_alpha <= 0:
        raise ValueError("max-length must be >= 16 and LoRA r/alpha must be positive.")

    LOG.info("Loading tokenizer and base model: %s", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_records = read_records(args.train_file, args.limit)
    rendered = [render_record(record, tokenizer, index) for index, record in enumerate(raw_records, start=1)]
    dataset = Dataset.from_dict({"text": rendered})

    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length, padding=False)

    tokenized = dataset.map(tokenize_batch, batched=True, remove_columns=["text"])
    if len(tokenized) == 0:
        raise ValueError("No training examples remain after tokenization.")
    LOG.info("Validated %d training records.", len(tokenized))

    model_kwargs: dict[str, Any] = {"torch_dtype": "auto"}
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_steps=args.save_steps,
        save_strategy="steps",
        save_total_limit=2,
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=True,
        fp16=torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 8,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        dataloader_pin_memory=torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        processing_class=tokenizer,
    )
    trainer.train()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    (args.output_dir / "ron1-training-metadata.json").write_text(
        json.dumps({
            "base_model": args.base_model,
            "training_records": len(tokenized),
            "epochs": args.epochs,
            "max_length": args.max_length,
            "method": "LoRA / PEFT",
            "runtime_note": "This adapter is not directly loadable by the browser ONNX runtime; merge, export, quantize, and test before deployment.",
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    LOG.info("Saved LoRA adapter to %s", args.output_dir)


if __name__ == "__main__":
    main()
