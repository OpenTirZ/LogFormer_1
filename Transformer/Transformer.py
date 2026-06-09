import torch
import torch.nn as nn

class TransformerBlock(nn.Module) :
    def __init__(self , cfg):
        super().__init__()

        self.attn = MultiHeadAttention(
            cfg["emb_dim"],
            cfg["emb_dim"] ,
            cfg["context_length"] ,
            cfg["drop_rate"] ,
            cfg["n_heads"] ,
            cfg["qkv_bias"]
        )

        self.ff = FeedForward(cfg)
        self.norm_1 = LayerNorm(cfg["emb_dim"])
        self.norm_2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self , x) :
        shortcut = x
        x = self.norm_1(x)
        x = self.attn(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm_2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x
