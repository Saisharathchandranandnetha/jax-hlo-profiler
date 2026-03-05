import optax


def create_optimizer():
    optimizer = optax.adam(learning_rate=3e-4)
    return optimizer
