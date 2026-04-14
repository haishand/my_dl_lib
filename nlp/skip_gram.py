import numpy as np

from common.layers import MatMul, SoftmaxWithLoss


class Skip_Gram:
    def __init__(self, vocab_size, hidden_size):
        V, H = vocab_size, hidden_size

        W_in = 0.01 * np.random.randn(V, H).astype("f")
        W_out = 0.01 * np.random.randn(H, V).astype("f")

        self.in_layer = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer0 = SoftmaxWithLoss()
        self.loss_layer1 = SoftmaxWithLoss()

        self.layers = [self.in_layer, self.out_layer]

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

        self.word_vec = W_in

    def forward(self, context, targets):
        h = self.in_layer.forward(context)
        out = self.out_layer.forward(h)
        loss0 = self.loss_layer0.forward(out, targets[:, 0, :])
        loss1 = self.loss_layer1.forward(out, targets[:, 1, :])
        loss = loss0 + loss1

        return loss

    def backward(self, dout=1):
        dl0 = self.loss_layer0.backward(dout)
        dl1 = self.loss_layer1.backward(dout)
        ds = dl0 + dl1
        dh = self.out_layer.backward(ds)
        self.in_layer.backward(dh)
        return None
