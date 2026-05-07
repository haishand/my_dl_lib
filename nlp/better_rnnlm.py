from common.base_model import BaseModel
from common.xp import *
from nlp.time_layer import (
    TimeAffine,
    TimeDropout,
    TimeEmbedding,
    TimeLSTM,
    TimeSoftmaxWithLoss,
)

"""
进行了如下优化：
1. LSTM多层化(两个)
2. dropout抑制过拟合
3. 权重共享(embedding和affine层权重共享)
"""


class BetterRnnLM(BaseModel):
    def __init__(self, vocab_size, wordvec_size, hidden_size, dropout_ratio=0.5):
        """Initialize the BetterRnnLM instance."""
        V, D, H = vocab_size, wordvec_size, hidden_size

        assert D == H, "权重共享必须让 wordvec_size == hidden_size (D == H)!"

        # xavier初始化
        embed_W = (rn(V, D) / np.sqrt(D)).astype("f")
        lstm_Wx1 = (rn(D, 4 * H) / np.sqrt(D)).astype("f")
        lstm_Wh1 = (rn(H, 4 * H) / np.sqrt(H)).astype("f")
        lstm_b1 = np.zeros(4 * H).astype("f")
        lstm_Wx2 = (rn(D, 4 * H) / np.sqrt(D)).astype("f")
        lstm_Wh2 = (rn(H, 4 * H) / np.sqrt(H)).astype("f")
        lstm_b2 = np.zeros(4 * H).astype("f")
        affine_b = np.zeros(V).astype("f")

        self.layers = [
            TimeEmbedding(embed_W),
            TimeDropout(dropout_ratio),
            TimeLSTM(lstm_Wx1, lstm_Wh1, lstm_b1, stateful=True),
            TimeDropout(dropout_ratio),
            TimeLSTM(lstm_Wx2, lstm_Wh2, lstm_b2, stateful=True),
            TimeDropout(dropout_ratio),
            TimeAffine(embed_W.T, affine_b),
        ]

        self.loss_layer = TimeSoftmaxWithLoss()
        self.lstm_layers = [self.layers[2], self.layers[4]]
        self.drop_layers = [self.layers[1], self.layers[3], self.layers[5]]

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def train(self):
        """Switch to training behavior."""
        for layer in self.drop_layers:
            layer.train_flg = True

    def test(self):
        """Switch to evaluation behavior."""
        for layer in self.drop_layers:
            layer.train_flg = False

    def predict(self, xs):
        """Predict the next word indices given input word indices."""
        self.test()
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs

    def forward(self, xs, ts):
        """Compute the loss for the given input and target word indices."""
        ys = self.predict(xs)
        loss = self.loss_layer.forward(ys, ts)
        return loss

    def backward(self, dout=1):
        """Perform backpropagation and compute gradients."""
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self):
        """Reset the hidden states of the LSTM layers."""
        for layer in self.lstm_layers:
            layer.reset_state()
