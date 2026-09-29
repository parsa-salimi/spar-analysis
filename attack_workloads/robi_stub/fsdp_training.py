"""TEST STUB: the model definition from workloads/fsdp_training.py on
multinode-tpu-experiments, verbatim, without the FSDP training loop."""
import torch
import torch.nn as nn

MODEL_CONFIGS = {
    "1b":  (2048, 24, 16, 8192,  32000),
    "7b":  (4096, 32, 32, 11008, 32000),
    "13b": (5120, 40, 40, 13824, 32000),
    "30b": (6656, 60, 52, 17920, 32000),
    "70b": (8192, 80, 64, 28672, 32000),
}


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, ffn_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim, bias=False),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_dim, bias=False),
        )

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class RandomTransformerLM(nn.Module):
    def __init__(self, config_name):
        super().__init__()
        hidden, n_layers, n_heads, ffn, vocab = MODEL_CONFIGS[config_name]
        self.vocab_size = vocab
        self.tok_emb = nn.Embedding(vocab, hidden)
        self.pos_emb = nn.Embedding(2048, hidden)
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden, n_heads, ffn) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.lm_head(x)
