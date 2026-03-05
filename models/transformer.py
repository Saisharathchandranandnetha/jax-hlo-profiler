import flax.linen as nn
from models.transformer_block import TransformerBlock


class TransformerModel(nn.Module):
    vocab_size: int
    hidden_dim: int
    num_heads: int
    num_layers: int

    def setup(self):
        self.embed = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.hidden_dim
        )

        self.layers = [
            TransformerBlock(self.hidden_dim, self.num_heads)
            for _ in range(self.num_layers)
        ]

        self.ln = nn.LayerNorm()
        self.head = nn.Dense(self.vocab_size)

    def __call__(self, x):
        x = self.embed(x)

        for layer in self.layers:
            x = layer(x)

        x = self.ln(x)
        logits = self.head(x)

        return logits
