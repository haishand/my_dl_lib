import os
import sys

sys.path.append(os.getcwd())
from nlp.RNN import RNN
import numpy as np

def test_rnn_forward_shape():
    D = 5
    H = 3
    N = 2

    Wx = np.random.randn(D, H)
    Wh = np.random.randn(H, H)
    b = np.random.randn(1, H)

    rnn = RNN(Wx, Wh, b)

    x = np.random.randn(N, D)
    h_prev = np.random.randn(N, H)

    h_next = rnn.forward(x, h_prev)

    assert h_next.shape == (N, H)

def test_rnn_backward_shape():
    D = 5
    H = 3
    N = 2

    Wx = np.random.randn(D, H)
    Wh = np.random.randn(H, H)
    b = np.random.randn(1, H)

    rnn = RNN(Wx, Wh, b)

    x = np.random.randn(N, D)
    h_prev = np.random.randn(N, H)

    h_next = rnn.forward(x, h_prev)

    dh_next = np.random.randn(N, H)
    dx, dh_prev = rnn.backward(dh_next)

    assert dx.shape == (N, D)
    assert dh_prev.shape == (N, H)