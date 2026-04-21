from re import X
from tkinter import Y

from common.xp import *
from common.functions import softmax, cross_entropy_error
from common.basic_layers import *


class AddLayer:
    def __init__(self):
        self.params, self.grads = [], []

    def forward(self, x, y):
        out = x + y
        return out

    def backward(self, dout=1):
        dx = dout * 1
        dy = dout * 1
        return dx, dy


class MulLayer:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None

    def forward(self, x, y):
        self.cache = (x, y)
        out = x * y
        return out

    def backward(self, dout=1):
        x, y = self.cache
        dx = dout * y
        dy = dout * x
        return dx, dy


class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.cache = None

    def forward(self, x):
        (W,) = self.params
        self.cache = x
        out = np.dot(x, W)
        return out

    def backward(self, dout=1):
        (W,) = self.params
        x = self.cache
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        self.grads[0][...] = dW
        return dx


class TanhLayer:
    def __init__(self):
        # 激活层无参数
        self.params = []
        self.grads = []
        self.cache = None

    def forward(self, x):
        # 前向
        y = np.tanh(x)
        self.cache = y  # 缓存输出，反向要用
        return y

    def backward(self, dout):
        y = self.cache
        # tanh 导数: dy/dx = 1 - y²
        dx = dout * (1 - y**2)
        return dx


class Sigmoid:
    def __init__(self):
        """Initialize sigmoid layer state."""
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        """Apply sigmoid activation."""
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        """Compute sigmoid backward pass."""
        dx = dout * (1.0 - self.out) * self.out
        return dx


class MyAffine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.matmul = MatMul(W)
        self.add = Add()
        self.x = None

    def forward(self, x):
        self.x = x
        out = self.matmul.forward(x)
        out = self.add.forward(out, self.params[1])
        return out

    def backward(self, dout):
        dout, db = self.add.backward(dout)
        dx = self.matmul.backward(dout)
        self.grads[0][...] = self.matmul.grads[0]
        self.grads[1][...] = db
        return dx


class Affine:
    def __init__(self, W, b):
        """Initialize affine layer parameters and buffers."""
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        """Apply linear transformation."""
        W, b = self.params
        out = np.dot(x, W) + b
        self.x = x
        return out

    def backward(self, dout):
        """Compute affine layer gradients."""
        W, b = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)

        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        """Initialize softmax-with-loss layer state."""
        self.params, self.grads = [], []
        self.loss = None
        self.y = None  # softmax的输出
        self.t = None  # 监督数据

    def forward(self, x, t):
        """Compute softmax output and cross-entropy loss."""
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        """Backpropagate gradient from softmax loss."""
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:  # 监督数据是one-hot-vector的情况
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx
