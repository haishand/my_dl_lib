import numpy as np

class TimeRNN:
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [
            np.zeros_like(Wx),
            np.zeros_like(Wh),
            np.zeros_like(b)]
        self.layer = None
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
        hs = np.empty((N, T, H), dtype='f')

        