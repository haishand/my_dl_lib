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
