import sys

sys.path.append("..")

import numpy as np
from common.layers import MatMul, AddLayer, TanhLayer


class RNN:
    def __init__(self, Wx, Wh, b) -> None:
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]

        self.mat_h = MatMul(Wh)
        self.mat_x = MatMul(Wx)
        self.add_1 = AddLayer()
        self.add_2 = AddLayer()
        self.tanh = TanhLayer()

        self.cache = None

    def forward(self, x, h_prev):
        Wx, Wh, b = self.params

        h_Wh = self.mat_h.forward(h_prev)
        x_Wx = self.mat_x.forward(x)
        sum1 = self.add_1.forward(h_Wh, x_Wx)
        sum2 = self.add_2.forward(sum1, b)
        h_next = self.tanh.forward(sum2)
        return h_next

    def backward(self, dh_next):
        Wx, Wh, b = self.params

        dsum2 = self.tanh.backward(dh_next)
        dsum1, db = self.add_2.backward(dsum2)
        dh_Wh, dx_Wx = self.add_1.backward(dsum1)
        dx = self.mat_x.backward(dx_Wx)
        dh_prev = self.mat_h.backward(dh_Wh)

        self.grads[0][...] = self.mat_x.grads[0]
        self.grads[1][...] = self.mat_h.grads[0]
        self.grads[2][...] = np.sum(db, axis=0)
        return dx, dh_prev


# test code
N, D, H = 2, 3, 4
Wx = np.random.randn(D, H)
Wh = np.random.randn(H, H)
b = np.random.randn(1, H)

rnn = RNN(Wx, Wh, b)
x = np.random.randn(N, D)
h_prev = np.random.randn(N, H)

h_next = rnn.forward(x, h_prev)
dx, dh = rnn.backward(np.ones_like(h_next))

print(h_next.shape)
print(dx.shape)
