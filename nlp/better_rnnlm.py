from common.xp import *
from nlp.time_layer import TimeEmbedding

"""
进行了如下优化：
1. LSTM多层化
2. dropout抑制过拟合
3. 权重共享(embedding和affine层权重共享)
"""


class BetterRnnLM:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        embed_W = (rn(V, D) / 100).astype("f")
        self.layers = [TimeEmbedding(W)]
