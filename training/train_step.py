import jax
import optax
from training.loss import cross_entropy


def create_train_step(model, optimizer):
    @jax.jit
    def train_step(params, opt_state, batch):
        def loss_fn(params):
            logits = model.apply({'params': params}, batch['input'])
            loss = cross_entropy(logits, batch['target'])
            return loss

        grads = jax.grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        return params, opt_state

    return train_step
