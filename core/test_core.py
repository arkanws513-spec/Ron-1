"""Basic shape and checkpoint tests for Ron-1 Native Core."""
import tempfile
import unittest
from pathlib import Path
import torch
from ron1_core import ByteTokenizer, Ron1Core, load_config, parameter_count

class Ron1CoreTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.config.update({"d_model": 32, "n_heads": 4, "n_layers": 1, "d_ff": 64, "max_seq_len": 32})
        torch.manual_seed(7)
        self.model = Ron1Core(self.config)

    def test_arabic_utf8_round_trip(self):
        tok = ByteTokenizer()
        text = "مرحبًا يا رون"
        self.assertEqual(tok.decode(tok.encode(text, add_bos=False)), text)

    def test_logits_and_loss(self):
        x = torch.randint(3, 259, (2, 8))
        y = torch.randint(3, 259, (2, 8))
        logits, loss = self.model(x, y)
        self.assertEqual(tuple(logits.shape), (2, 8, 259))
        self.assertTrue(torch.isfinite(loss).item())

    def test_weights_are_trainable_and_tied(self):
        self.assertGreater(parameter_count(self.model), 0)
        self.assertIs(self.model.lm_head.weight, self.model.token_embedding.weight)

    def test_autoregressive_generation_shape(self):
        x = torch.tensor([[1, 77, 78]], dtype=torch.long)
        out = self.model.generate(x, max_new_tokens=2, temperature=0)
        self.assertEqual(tuple(out.shape), (1, 5))

if __name__ == "__main__":
    unittest.main()
