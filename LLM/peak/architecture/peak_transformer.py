import torch.nn as nn
from .peak_multi_head_attention import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout=0.1):
        super().__init__()

        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embed_dim)
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention + residual
        x = self.norm1(x)
        attn_out = self.attn(x, mask)
        x = x + self.dropout(attn_out)
        # Feedforward + residual
        x = self.norm2(x)
        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        return x