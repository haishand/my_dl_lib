import os
import sys

sys.path.append(os.getcwd())

import numpy as np
from nlp.clip_grads import clip_grads

dW1 = np.random.rand(3, 3) * 10
dW2 = np.random.rand(3, 3) * 10
grads = [dW1, dW2]
max_norm = 5.0

def calc_norm(grads):
    """Execute calc_norm."""
    norm = 0
    for g in grads:
        norm += np.sum(g**2)
    norm = np.sqrt(norm)
    return norm

old_norm = calc_norm(grads)
clip_grads(grads, max_norm)
new_norm = calc_norm(grads)
assert old_norm >= new_norm
