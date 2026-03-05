import flax.linen as nn
import jax.numpy as jnp


class TransformerBlock(nn.Module):
    hidden_dim: int
    num_heads: int

    def setup(self):
        self.attn = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_dim
        )

        self.mlp = nn.Sequential([
            nn.Dense(self.hidden_dim * 4),
            nn.gelu,
            nn.Dense(self.hidden_dim)
        ])

        self.ln1 = nn.LayerNorm()
        self.ln2 = nn.LayerNorm()

    def __call__(self, x):
        h = self.ln1(x)
        x = x + self.attn(h)

        h = self.ln2(x)
        x = x + self.mlp(h)

        return x
