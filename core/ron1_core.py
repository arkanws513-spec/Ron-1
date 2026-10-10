"""Ron-1 native decoder-only Transformer. No pretrained model dependency."""
import json
from pathlib import Path
from typing import Iterable
import torch
from torch import nn
from torch.nn import functional as F

CORE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = CORE_DIR / "config.json"

def load_config(path=CONFIG_PATH):
    return json.loads(Path(path).read_text(encoding="utf-8"))

class ByteTokenizer:
    """PAD=0, BOS=1, EOS=2, UTF-8 bytes=3..258."""
    pad_id, bos_id, eos_id, vocab_size = 0, 1, 2, 259
    def encode(self, text: str, add_bos=True, add_eos=False):
        ids = [b + 3 for b in text.encode("utf-8", errors="replace")]
        if add_bos: ids.insert(0, self.bos_id)
        if add_eos: ids.append(self.eos_id)
        return ids
    def decode(self, ids: Iterable[int]):
        return bytes(int(i)-3 for i in ids if 3 <= int(i) <= 258).decode("utf-8", errors="replace")

class CausalSelfAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        dim, heads = c["d_model"], c["n_heads"]
        if dim % heads: raise ValueError("d_model must be divisible by n_heads")
        self.n_heads, self.head_dim = heads, dim // heads
        self.qkv = nn.Linear(dim, 3*dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.register_buffer("causal_mask", torch.tril(torch.ones(c["max_seq_len"], c["max_seq_len"], dtype=torch.bool)), persistent=False)
    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def split(t): return t.view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        q, k, v = split(q), split(k), split(v)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=self.causal_mask[:n, :n], dropout_p=0.0, is_causal=False)
        return self.proj(y.transpose(1, 2).contiguous().view(b, n, d))

class MLP(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.up, self.down = nn.Linear(dim, hidden, bias=False), nn.Linear(hidden, dim, bias=False)
    def forward(self, x): return self.down(F.gelu(self.up(x)))

class TransformerBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        d = c["d_model"]
        self.ln1, self.attn = nn.LayerNorm(d), CausalSelfAttention(c)
        self.ln2, self.mlp = nn.LayerNorm(d), MLP(d, c["d_ff"])
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))

class Ron1Core(nn.Module):
    """Ron-1's own trainable model parameters."""
    def __init__(self, config=None):
        super().__init__()
        self.config = config or load_config()
        c = self.config
        self.token_embedding = nn.Embedding(c["vocab_size"], c["d_model"])
        self.position_embedding = nn.Embedding(c["max_seq_len"], c["d_model"])
        self.blocks = nn.ModuleList([TransformerBlock(c) for _ in range(c["n_layers"])])
        self.final_norm = nn.LayerNorm(c["d_model"])
        self.lm_head = nn.Linear(c["d_model"], c["vocab_size"], bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)
    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)): nn.init.normal_(m.weight, mean=0.0, std=0.02)
    def forward(self, input_ids, targets=None):
        b, n = input_ids.shape
        if n > self.config["max_seq_len"]: raise ValueError("input exceeds max_seq_len")
        pos = torch.arange(n, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(pos)[None, :, :]
        for block in self.blocks: x = block(x)
        logits = self.lm_head(self.final_norm(x))
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100) if targets is not None else None
        return logits, loss
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=80, temperature=0.8, top_k=40):
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(input_ids[:, -self.config["max_seq_len"]:])
            logits = logits[:, -1, :]
            if temperature <= 0: nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k > 0:
                    k = min(top_k, logits.size(-1))
                    logits = logits.masked_fill(logits < torch.topk(logits, k).values[:, -1, None], float("-inf"))
                nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            input_ids = torch.cat((input_ids, nxt), dim=1)
            if int(nxt.item()) == ByteTokenizer.eos_id: break
        return input_ids

def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
