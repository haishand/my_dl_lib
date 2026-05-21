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

        self.embed_layer = TimeEmbedding(embed_W)
        self.lstm_layer = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=False)
        self.layers = [self.embed_layer]
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


class PeekyDecoder(BaseModel):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size

        embed_W = (rn(V, D) / 100).astype("f")
        lstm_Wx = (rn(H + D, 4 * H) / np.sqrt(H + D)).astype("f")
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype("f")
        lstm_b = np.zeros(4 * H)
        affine_W = (rn(H + H, V) / np.sqrt(2 * H)).astype("f")
        affine_b = np.zeros(V)
        self.cache = None

        self.embed_layer = TimeEmbedding(embed_W)
        self.lstm_layer = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.affine_layer = TimeAffine(affine_W, affine_b)
        self.layers = [
            self.embed_layer,
            self.lstm_layer,
            self.affine_layer,
        ]

        super().__init__(self.layers)

    def forward(self, xs, h):
        """
        xs.shape: (N, T) 这个和encoder里的xs不一样
        h.shape: (N, H)
        """
        self.lstm_layer.set_state(h)

        N, T = xs.shape
        N, H = h.shape
        hs = np.repeat(h, T, axis=0).reshape(N, T, H)
        out = self.embed_layer.forward(xs)
        out = np.concatenate((hs, out), axis=2)
        out = self.lstm_layer.forward(out)
        out = np.concatenate((hs, out), axis=2)
        score = self.affine_layer.forward(out)

        self.cache = H
        return score

    def backward(self, dout):
        dout = self.affine_layer.backward(dout)
        H = self.cache
        dout, dhs0 = dout[:, :, H:], dout[:, :, :H]
        dout = self.lstm_layer.backward(dout)
        dembed, dhs1 = dout[:, :, H:], dout[:, :, :H]
        self.embed_layer.backward(dembed)

        dhs = dhs0 + dhs1
        dh = self.lstm_layer.dh + np.sum(dhs, axis=1)
        return dh

    def generate(self, h, start_id, sample_size):
        self.lstm_layer.set_state(h)
        sample = []
        sample_id = start_id

        for _ in range(sample_size):
            #            x = np.array(sample_id).reshape((1, 1))
            x = np.array([[sample_id]])  # 强制保证 (1,1) 二维

            N, T = x.shape
            N, H = h.shape

            hs = np.repeat(h, T, axis=0).reshape(N, T, H)
            out = self.embed_layer.forward(x)
            out = np.concatenate((hs, out), axis=2)
            out = self.lstm_layer.forward(out)
            out = np.concatenate((hs, out), axis=2)
            score = self.affine_layer.forward(out)

            sample_id = np.argmax(score.flatten())
            sample.append(int(sample_id))
        return sample


class PeekySeq2seq(BaseModel):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        self.encoder = Encoder(V, D, H)
        self.decoder = PeekyDecoder(V, D, H)
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
