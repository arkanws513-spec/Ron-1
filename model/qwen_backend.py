from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from model.prepare_model import ensure_local_model


class QwenBackend:
    """Loads the Ron-1 local Qwen3-1.7B model weights."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        if self.model is not None:
            return

        local_path = ensure_local_model()
        self.tokenizer = AutoTokenizer.from_pretrained(
            local_path,
            local_files_only=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            local_path,
            torch_dtype="auto",
            device_map="auto",
            local_files_only=True,
        )

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        self.load()

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=top_p,
            )

        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip()
