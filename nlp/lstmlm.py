from common.xp import *

import pickle

from common.base_model import BaseModel
from nlp.time_layer import *


class LSTMLM(BaseModel):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        super().__init__()

        V, D, H = vocab_size, wordvec_size, hidden_size

        # 初始化权重
        embed_W = (np.random.randn(V, D) / 100).astype("f")
        lstm_Wx = (np.random.randn(D, 4 * H) / np.sqrt(D)).astype("f")
        lstm_Wh = (np.random.randn(H, 4 * H) / np.sqrt(H)).astype("f")
        lstm_b = np.zeros(4 * H).astype("f")
        affine_W = (np.random.randn(H, V) / np.sqrt(H)).astype("f")
        affine_b = np.zeros(V).astype("f")

        # 创建层
        self.layers = [
            TimeEmbedding(embed_W),
            TimeLSTM(lstm_Wx, lstm_Wh, lstm_b),
            TimeAffine(affine_W, affine_b),
        ]

        self.loss_layer = TimeSoftmaxWithLoss()

        self.lstm_layer = self.layers[1]

        # 保存权重和梯度
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs, ts):
        for layer in self.layers:
            xs = layer.forward(xs)
        loss = self.loss_layer.forward(xs, ts)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self):
        self.lstm_layer.reset_state()
