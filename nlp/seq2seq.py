import numpy as np
from time_layer import *
from common.base_model import BaseModel


class Encoder(BaseModel):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size

        embed_W = (rn(V, D) / 100).astype("f")
        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype("f")
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype("f")
        lstm_b = np.zeros(4 * H)

        self.layers = [
            TimeEmbedding(embed_W),
            TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=False),
        ]
        super().__init__(self.layers)
        self.hs = None

    def forward(self, xs):
        for layer in self.layers:
            xs = layer.forward(xs)
        self.hs = xs
        return self.hs[:, -1, :]

    def backward(self, dh):
        dout = np.zeros_like(self.hs)
        dout[:, -1, :] = dh

        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout


class Decoder(BaseModel):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size

        embed_W = (rn(V, D) / 100).astype("f")
        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype("f")
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype("f")
        lstm_b = np.zeros(4 * H)
        affine_W = (rn(H, V) / np.sqrt(H)).astype("f")
        affine_b = np.zeros(V)

        self.layers = [
            TimeEmbedding(embed_W),
            TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True),
            TimeAffine(affine_W, affine_b),
        ]

        self.lstm_layer = self.layers[1]
        super().__init__(self.layers)

    def forward(self, xs, h):
        self.lstm_layer.set_state(h)
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        dh = self.lstm_layer.dh
        return dh

    def generate(self, h, start_id, sample_size):
        self.lstm_layer.set_state(h)
        sample = []
        sample_id = start_id

        for _ in range(sample_size):
            #            x = np.array(sample_id).reshape((1, 1))
            x = np.array([[sample_id]])  # 强制保证 (1,1) 二维
            for layer in self.layers:
                x = layer.forward(x)
            sample_id = np.argmax(x.flatten())
            sample.append(int(sample_id))
        return sample


class Seq2seq(BaseModel):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        self.encoder = Encoder(V, D, H)
        self.decoder = Decoder(V, D, H)
        super().__init__(self.encoder.layers + self.decoder.layers)
        self.loss_layer = TimeSoftmaxWithLoss()

    def forward(self, xs, ts):
        decode_xs, decode_ts = ts[:, :-1], ts[:, 1:]

        h = self.encoder.forward(xs)
        score = self.decoder.forward(decode_xs, h)
        loss = self.loss_layer.forward(score, decode_ts)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        dh = self.decoder.backward(dout)
        dout = self.encoder.backward(dh)
        return dout

    def generate(self, xs, start_id, sample_size):
        h = self.encoder.forward(xs)
        sample = self.decoder.generate(h, start_id, sample_size)
        return sample
