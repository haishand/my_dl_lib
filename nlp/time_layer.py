import numpy as np
from common.layers import (
    MatMul,
    AddLayer,
    SoftmaxWithLoss,
    TanhLayer,
    Affine,
    MyAffine,
    Embedding,
)


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


class TimeRNN:
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None
        self.h, self.dh = None, None
        self.stateful = stateful

    def set_state(self, h):
        self.h = h

    def reset_state(self):
        self.h = None

    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        D, H = Wx.shape

        self.layers = []
        hs = np.empty((N, T, H), dtype="f")

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype="f")

        for t in range(T):
            layer = RNN(*self.params)
            self.h = layer.forward(xs[:, t, :], self.h)
            hs[:, t, :] = self.h
            self.layers.append(layer)
        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D, H = Wx.shape

        dxs = np.empty((N, T, D), dtype="f")
        dh = np.zeros_like(self.h)
        grads = [0, 0, 0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh = layer.backward(dhs[:, t, :] + dh)
            dxs[:, t, :] = dx

            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh
        return dxs


class TimeAffine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.layers = None
        self.x = None

    def forward(self, xs):
        N, T, D = xs.shape
        W, b = self.params

        rx = xs.reshape(N * T, D)
        out = np.dot(rx, W) + b
        self.x = xs
        return out.reshape(N, T, -1)

    def backward(self, dout):
        xs = self.x
        N, T, D = xs.shape
        W, b = self.params

        dout_reshaped = dout.reshape(N * T, -1)
        rx = xs.reshape(N * T, D)
        dW = np.dot(rx.T, dout_reshaped)
        db = np.sum(dout_reshaped, axis=0)
        dx = np.dot(dout_reshaped, W.T)

        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx.reshape(N, T, D)


class TimeEmbedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.layers = None

    def forward(self, xs):
        (W,) = self.params
        N, T = xs.shape
        V, D = W.shape
        out = np.empty((N, T, D), dtype="f")
        self.layers = []
        for t in range(T):
            layer = Emb
            edding(W)
            out[:, t, :] = layer.forward(xs[:, t])
            self.layers.append(layer)
        return out

    def backward(self, dout):
        N, T, D = dout.shape

        grads = 0
        for t in range(T):
            self.layers[t].backward(dout[:, t, :])
            grads += self.layers[t].grads[0]
        self.grads[0][...] = grads
        return None


class TimeSoftmaxWithLoss:
    def __init__(self):
        self.layers = None
        self.cache = None

    def forward(self, xs, ts):
        N, T, V = xs.shape
        self.cache = (N, T, V)

        self.layers = []
        loss = 0.0
        for t in range(T):
            layer = SoftmaxWithLoss()
            loss += layer.forward(xs[:, t, :], ts[:, t])
            self.layers.append(layer)
        return loss / T

    def backward(self, dout=1):
        N, T, V = self.cache
        dxs = np.empty((N, T, V), dtype='f')
        for t in range(T):
            t = T-t-1
            layer = self.layers[t]
            dxs[:, t, :] = layer.backward(dout)

        return dxs

