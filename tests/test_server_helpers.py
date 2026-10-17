import unittest

import torch

from core import server


class DummyTokenizer:
    chat_template = None

    def __init__(self):
        self.seen_text = ""

    def __call__(self, text, return_tensors, truncation, max_length):
        self.seen_text = text
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class ServerHelperTests(unittest.TestCase):
    def test_history_discards_invalid_roles_and_bounds_entries(self):
        history = [
            {"role": "system", "content": "ignore"},
            {"role": "user", "content": "  مرحبًا  "},
            {"role": "assistant", "content": "أهلًا"},
            {"role": "user", "content": 42},
        ]
        safe = server.sanitize_history(history, "سؤال جديد")
        self.assertEqual(
            safe,
            [
                {"role": "user", "content": "مرحبًا"},
                {"role": "assistant", "content": "أهلًا"},
                {"role": "user", "content": "سؤال جديد"},
            ],
        )

    def test_current_user_message_is_not_duplicated(self):
        history = [{"role": "user", "content": "ما عاصمة مصر؟"}]
        safe = server.sanitize_history(history, "ما عاصمة مصر؟")
        self.assertEqual(len(safe), 1)

    def test_history_is_limited_to_recent_entries(self):
        history = [{"role": "user", "content": str(i)} for i in range(20)]
        safe = server.sanitize_history(history, "current")
        self.assertLessEqual(len(safe), 13)
        self.assertEqual(safe[-1]["content"], "current")

    def test_fallback_prompt_builder_limits_input_length(self):
        tokenizer = DummyTokenizer()
        original_limit = server.MAX_INPUT_TOKENS
        try:
            server.MAX_INPUT_TOKENS = 3
            input_ids, attention_mask = server.build_inputs(
                tokenizer, "اختبار", [{"role": "user", "content": "اختبار"}]
            )
            self.assertEqual(input_ids.shape[-1], 3)
            self.assertEqual(attention_mask.shape[-1], 3)
            self.assertIn("اختبار", tokenizer.seen_text)
        finally:
            server.MAX_INPUT_TOKENS = original_limit


if __name__ == "__main__":
    unittest.main()
