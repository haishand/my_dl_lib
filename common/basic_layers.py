import numpy as np


class Add:
    def __init__(self):
        """Initialize addition layer state."""
        self.params, self.grads = [], []
        self.x0 = None
        self.x1 = None

    def forward(self, x0, x1):
        """Apply element-wise addition."""
        self.x0 = x0
        self.x1 = x1
        out = x0 + x1
        return out

    def backward(self, dout):
        """Compute addition layer gradients."""
        dx0 = dout * np.ones_like(self.x0)
        dx1 = dout * np.ones_like(self.x1)
        dx1 = np.sum(dx1, axis=0, keepdims=True)
        return dx0, dx1


class Mul:
    def __init__(self):
        """Initialize multiplication layer state."""
        self.params, self.grads = [], []
        self.x0 = None
        self.x1 = None

    def forward(self, x0, x1):
        """Apply element-wise multiplication."""
        self.x0 = x0
        self.x1 = x1
        out = x0 * x1
        return out

    def backward(self, dout):
        """Compute multiplication layer gradients."""
        dx0 = dout * np.ones_like(self.x0)
        dx1 = dout * np.ones_like(self.x1)
        return dx0, dx1


class MatMul:
    def __init__(self, W):
        """Initialize matrix multiplication layer parameters and buffers."""
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None

    def forward(self, x):
        """Apply matrix multiplication."""
        W = self.params[0]
        out = np.dot(x, W)
        self.x = x
        return out

    def backward(self, dout):
        """Compute matrix multiplication layer gradients."""
        W = self.params[0]
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)

        self.grads[0][...] = dW
        return dx


class Repeat:
    def __init__(self, times):
        """Initialize repeat layer state."""
        self.params, self.grads = [], []
        self.times = times
        self.x = None

    def forward(self, x):
        """Apply repeat operation."""
        self.x = x
        out = np.repeat(x, self.times, axis=0)
        return out

    def backward(self, dout):
        """Compute repeat layer gradients."""
        dx = np.sum(dout, axis=0, keepdims=True)
        return dx


class Sum:
    def __init__(self):
        """Initialize sum layer state."""
        self.params, self.grads = [], []
        self.x = None

    def forward(self, x):
        """Apply sum operation."""
        self.x = x
        out = np.sum(x, axis=0, keepdims=True)
        return out

    def backward(self, dout):
        """Compute sum layer gradients."""
        dx = np.repeat(dout, self.x.shape[0], axis=0)
        return dx
