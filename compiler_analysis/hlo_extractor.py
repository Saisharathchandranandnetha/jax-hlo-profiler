import jax


def extract_hlo(train_step, params, opt_state, batch):
    """
    Extract HLO (High-Level Optimizer) IR from a JAX function.
    Uses the modern JAX API: jit().lower().compiler_ir()
    """
    # Create a lowered representation of the JIT-compiled function
    lowered = jax.jit(train_step).lower(params, opt_state, batch)
    
    # Get the HLO module text representation
    hlo_text = lowered.compiler_ir(dialect='hlo').as_hlo_text()
    
    return hlo_text


def save_hlo(hlo_text, path):
    """
    Save HLO text to a file.
    """
    with open(path, 'w', encoding='utf-8') as f:
        f.write(hlo_text)
    
    print(f'HLO graph saved to {path}')
