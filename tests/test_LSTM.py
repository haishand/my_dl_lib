import os
import sys

sys.path.append(os.getcwd())

import numpy as np
import pytest

from common.functions import sigmoid
from nlp.time_layer import LSTM


rn = np.random.randn


# forward 输出形状的正确
def test_lstm_forward_shape():
    """Execute test_lstm_forward_shape."""
    D = 5
    H = 3
    N = 2

    Wx = rn(D, 4 * H)
    Wh = rn(H, 4 * H)
    b = rn(4 * H)

    lstm = LSTM(Wx, Wh, b)

    x = rn(N, D)
    h_prev = rn(N, H)
    c_prev = rn(N, H)

    h_next, c_next = lstm.forward(x, h_prev, c_prev)

    assert h_next.shape == (N, H)
    assert c_next.shape == (N, H)


# backward 输出形状正确
def test_lstm_backward_shape():
    """Execute test_lstm_backward_shape."""
    D = 5
    H = 3
    N = 2

    Wx = rn(D, 4 * H)
    Wh = rn(H, 4 * H)
    b = rn(4 * H)

    lstm = LSTM(Wx, Wh, b)

    x = rn(N, D)
    h_prev = rn(N, H)
    c_prev = rn(N, H)

    h_next, c_next = lstm.forward(x, h_prev, c_prev)

    dh = rn(N, H)
    dc = rn(N, H)
    dx, dh_prev, dc_prev = lstm.backward(dh, dc)

    assert dx.shape == (N, D)
    assert dh_prev.shape == (N, H)
    assert dc_prev.shape == (N, H)


# grads 不为0, 不为nan
def test_lstm_grads_not_zero_or_nan():
    """Execute test_lstm_grads_not_zero_or_nan."""
    D = 5
    H = 3
    N = 2

    Wx = rn(D, 4 * H)
    Wh = rn(H, 4 * H)
    b = rn(4 * H)

    lstm = LSTM(Wx, Wh, b)

    x = rn(N, D)
    h_prev = rn(N, H)
    c_prev = rn(N, H)

    h_next, c_next = lstm.forward(x, h_prev, c_prev)

    dh = rn(N, H)
    dc = rn(N, H)
    lstm.backward(dh, dc)

    for i in range(3):
        assert np.all(lstm.grads[i] != 0)
        assert not np.isnan(lstm.grads[i]).any()
        assert not np.isinf(lstm.grads[i]).any()
