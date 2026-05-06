from common.xp import *
from common.layers import (
    MatMul,
    AddLayer,
    SoftmaxWithLoss,
    TanhLayer,
    Affine,
    MyAffine,
    Embedding,
)
from common.functions import sigmoid


class RNN:
    """
    RNN单元
    """

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
    """
    一堆共享权重的RNN单元串起来处理时间序列组成了RNN层
    """

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
            layer = Embedding(W)
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
        dxs = np.empty((N, T, V), dtype="f")
        for t in range(T):
            t = T - t - 1
            layer = self.layers[t]
            dxs[:, t, :] = layer.backward(dout)

        return dxs


class LSTM:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev, c_prev):
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


class TimeLSTM:
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None
        self.h, self.c, self.dh = None, None, None
        self.stateful = stateful

    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        H = Wh.shape[0]

        self.layers = []
        hs = np.empty((N, T, H), dtype="f")

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype="f")
        if not self.stateful or self.c is None:
            self.c = np.zeros((N, H), dtype="f")

        for t in range(T):
            layer = LSTM(*self.params)
            self.h, self.c = layer.forward(xs[:, t, :], self.h, self.c)
            hs[:, t, :] = self.h
            self.layers.append(layer)
        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D, H = Wx.shape

        dxs = np.empty((N, T, D), dtype="f")
        dh = np.zeros_like(self.h)
        dc = np.zeros_like(self.c)
        grads = [0, 0, 0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh, dc = layer.backward(dhs[:, t, :] + dh, dc)
            dxs[:, t, :] = dx

            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh
        return dxs

    def set_state(self, h, c):
        self.h = h
        self.c = c

    def reset_state(self):
        self.h = None
        self.c = None


class TimeDropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.train_flg = True

    def forward(self, xs):
        if self.train_flg:  # 训练模式，执行dropout
            flg = rd(*xs.shape) > self.dropout_ratio

            scale = 1.0 / (1.0 - self.dropout_ratio)

            self.mask = flg.astype("f") * scale

            return xs * self.mask

        else:  # 测试模式，不做dropout，直接返回
            return xs

    def backward(self, dout):
        return dout * self.mask
