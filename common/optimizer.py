from common.xp import np


class SGD:
    """
    随机梯度下降法（Stochastic Gradient Descent）
    """

    def __init__(self, lr=0.01):
        """Initialize optimizer hyperparameters."""
        self.lr = lr

    def update(self, params, grads):
        """Update parameters using SGD."""
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]


class AdaGrad:
    def __init__(self, lr):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = []
            for param in params:
                self.h.append(np.zero_like(param))

        for i in range(len(params)):
            self.h[i] += grads[i] ** 2
            params[i] -= self.lr * grads[i] / (np.sqrt(self.h[i]) + 1e-7)
