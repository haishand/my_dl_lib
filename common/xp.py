# coding: utf-8
from common.config import GPU


if GPU:
    import cupy as np

    np.cuda.set_allocator(np.cuda.MemoryPool().malloc)

    print("\033[92m" + "-" * 60 + "\033[0m")
    print(" " * 23 + "\033[92mGPU Mode (cupy)\033[0m")
    print("\033[92m" + "-" * 60 + "\033[0m\n")
else:
    import numpy as np


# 生成 -∞ ~ +∞ 正态分布
rn = np.random.randn

# 生成 0 ~ 1 之间均匀分布
rd = np.random.rand
