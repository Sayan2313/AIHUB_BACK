import torch.nn as nn
import torch
from .sin_cos_pos_emb import SinCosPositionalEncoding
from .peak_transformer import TransformerBlock

class PeakModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_heads,
        num_layers,
        seq_length
    ):
        super().__init__()

        # Token embeddings (random init ✔)
        self.token_emb = nn.Embedding(vocab_size, embed_dim)

        # SinCos positional encoding ✔
        self.pos_enc = SinCosPositionalEncoding(embed_dim, seq_length)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, embed_dim * 4)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(embed_dim)

        # Output head (next token prediction ✔)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        B, T = x.shape

        # Token + position
        x = self.token_emb(x)
        x = self.pos_enc(x)

        # Causal mask (VERY IMPORTANT)
        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.final_norm(x)

        logits = self.lm_head(x)  # (B, T, vocab)

        return logits