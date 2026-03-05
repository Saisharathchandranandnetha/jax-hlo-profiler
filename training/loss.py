import jax
import jax.numpy as jnp


def cross_entropy(logits, targets):
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    one_hot = jnp.eye(logits.shape[-1])[targets]
    return -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))
