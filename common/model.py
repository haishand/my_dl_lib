from common.xp import *
from common.layers import Affine, Sigmoid, SoftmaxWithLoss


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        """Initialize a two-layer fully connected network."""
        I, H, O = input_size, hidden_size, output_size

        # 初始化权重和偏置
        W1 = np.random.randn(I, H)
        b1 = np.random.randn(H)
        W2 = np.random.randn(H, O)
        b2 = np.random.randn(O)

        # 生成层
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2),
        ]

        self.loss_layer = SoftmaxWithLoss()

        # 将所有权重整理到列表中
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, x):
        """Run forward inference and return logits."""
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def forward(self, x, t):
        """Compute the training loss for a batch."""
        score = self.predict(x)
        loss = self.loss_layer.forward(score, t)
        return loss

    def backward(self, dout=1):
        """Backpropagate gradients through all layers."""
        dout = self.loss_layer.backward(dout)

        for layer in reversed(self.layers):
            dout = layer.backward(dout)

        return dout
