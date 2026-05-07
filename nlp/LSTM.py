import numpy as np
from common.functions import sigmoid


class LSTM:
    def __init__(self, Wx, Wh, b):
        """Initialize the LSTM instance."""
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev, c_prev):
        """Run the forward pass."""
        Wx, Wh, b = self.params
        N, H = h_prev.shape

        A = np.dot(x, Wx) + np.dot(h_prev, Wh) + b

        # slice
        f = A[:, :H]
        g = A[:, H : 2 * H]
        i = A[:, 2 * H : 3 * H]
        o = A[:, 3 * H :]

        f = sigmoid(f)
        g = np.tanh(g)
        i = sigmoid(i)
        o = sigmoid(o)

        c_next = f * c_prev + g * i
        h_next = o * np.tanh(c_next)

        self.cache = (x, h_prev, c_prev, i, f, g, o, c_next)
        return h_next, c_next

    def backward(self, dh, dc):
        """Run the backward pass."""
        x, h_prev, c_prev, i, f, g, o, c_next = self.cache
        Wx, Wh, b = self.params

        dc = dc + dh * o * (1 - np.tanh(c_next) ** 2)
        do = dh * np.tanh(c_next)

        do = do * (1 - o) * o
        di = dc * g * (1 - i) * i
        dg = dc * i * (1 - g**2)
        df = dc * c_prev * (1 - f) * f

        dA = np.hstack((df, dg, di, do))

        dWx = np.dot(x.T, dA)
        dWh = np.dot(h_prev.T, dA)
        db = np.sum(dA, axis=0)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        dx = np.dot(dA, Wx.T)
        dh_prev = np.dot(dA, Wh.T)
        dc_prev = dc * f

        return dx, dh_prev, dc_prev
