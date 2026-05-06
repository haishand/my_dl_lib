import os
import sys

sys.path.append(os.getcwd())
from nlp.time_layer import *
import pytest

"""
词汇表 V = 5 个词：[我，爱，学，习，。]
句子长度 T = 3
批量 N = 2

"""
def test_TimeSoftmaxWithLoss_shape():
    layer = TimeSoftmaxWithLoss()

    ts = [
        [1, 2, 3],   # 句子1：爱 → 学 → 习
        [0, 1, 4]    # 句子2：我 → 爱 → 。
    ]
    xs = [
        # 句子1，3个时间步，每个步长5个概率
        [[0.1, 0.7, 0.1, 0.05, 0.05],  # t0: 预测1号词（爱）
        [0.05, 0.1, 0.8, 0.03, 0.02], # t1: 预测2号词（学）
        [0.01, 0.02, 0.04, 0.9, 0.03]] # t2: 预测3号词（习）
    ]
    ts =np.array(ts)
    xs = np.array(xs)
    loss = layer.forward(xs, ts)

    assert isinstance(loss, (int, float))

def test_TimeLSTM_forward_shape():
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

def test_TimeLSTM_backward_shape():
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