import jax
import jax.numpy as jnp

from models.transformer import TransformerModel
from training.optimizer import create_optimizer
from training.train_step import create_train_step
from compiler_analysis.hlo_extractor import extract_hlo, save_hlo


def main():

    model = TransformerModel(
        vocab_size=32000,
        hidden_dim=256,
        num_heads=4,
        num_layers=4
    )

    rng = jax.random.PRNGKey(0)

    dummy_input = jnp.ones((8, 128), dtype=jnp.int32)

    params = model.init(rng, dummy_input)["params"]

    optimizer = create_optimizer()

    opt_state = optimizer.init(params)

    train_step = create_train_step(model, optimizer)

    batch = {
        "input": dummy_input,
        "target": dummy_input
    }

    print("Extracting HLO graph...")

    hlo_text = extract_hlo(train_step, params, opt_state, batch)

    save_hlo(hlo_text, "results/transformer_train_step_hlo.txt")

    print("Done.")


if __name__ == "__main__":
    main()
