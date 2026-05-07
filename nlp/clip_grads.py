import numpy as np

def clip_grads(grads, max_norm):
    """Execute clip_grads."""
    total_norm = 0
    for g in grads:
        total_norm += np.sum(g**2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / total_norm
    if rate < 1:
        for g in grads:
            g *= rate
