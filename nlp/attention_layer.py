from common.xp import *
from common.base_model import BaseModel
from common.layers import Softmax
from nlp.time_layer import TimeEmbedding, TimeLSTM, TimeAffine
from seq2seq import Encoder, Decoder


class WeightSum(BaseModel):
    def __init__(self):
        self.cache = None

    def forward(self, hs, a):
        N, T, H = hs.shape
        N, T = a.shape

        ar = a.reshape(N, T, 1).repeat(H, axis=2)
        c = hs * ar
        c = np.sum(c, axis=1)

        self.cache = (hs, ar)
        return c

    def backward(self, dc):
        hs, ar = self.cache
        N, T, H = hs.shape

        dt = dc.reshape(N, 1, H).repeat(T, axis=1)
        dhs = ar * dt
        dar = hs * dt
        dar = np.sum(dar, axis=2)
        return dhs, dar


class AttentionWeight(BaseModel):
    def __init__(self):
        self.softmax_layer = Softmax()
        self.cache = None

    def forward(self, hs, h):
        N, T, H = hs.shape
        N, H = h.shape

        hr = h.reshape(N, 1, H).repeat(T, axis=1)
        out = hs * hr
        out = np.sum(out, axis=2)
        out = self.softmax_layer.forward(out)

        self.cache = (hs, hr)

        return out

    def backward(self, dout):
        hs, hr = self.cache
        N, T, H = hs.shape

        dout = self.softmax_layer.backward(dout)
        dout = dout.reshape(N, T, 1).repeat(H, axis=2)
        dhs = hr * dout
        dhr = hs * dout
        dh = np.sum(dhr, axis=1)
        return dhs, dh


class Attention(BaseModel):
    def __init__(self):
        self.attention_weight_layer = AttentionWeight()
        self.weight_sum_layer = WeightSum()
        self.attention_weight = None

    def forward(self, hs, h):
        a = self.attention_weight_layer.forward(hs, h)
        out = self.weight_sum_layer.forward(hs, a)
        self.attention_weight = a
        return out

    def backward(self, dout):
        dhs0, da = self.weight_sum_layer.backward(dout)
        dhs1, dh = self.attention_weight_layer.backward(da)
        dhs = dhs0 + dhs1
        return dhs, dh


class TimeAttention(BaseModel):
    def __init__(self):
        self.params, self.grads = [], []
        self.layers = None
        self.attention_weights = None

    def forward(self, hs_enc, hs_dec):
        N, T, H = hs_dec.shape

        self.layers = []
        self.attention_weights = []
        out = np.empty_like(hs_dec)
        for t in range(T):
            layer = Attention()
            out[:, t, :] = layer.forward(hs_enc, hs_dec[:, t, :])
            self.layers.append(layer)
            self.attention_weights.append(layer.attention_weight)
        return out

    def backward(self, dout):
        N, T, H = dout.shape
        dhs_enc = 0
        dhs_dec = np.empty_like(dout)
        for t in range(T):
            layer = self.layers[t]
            dhs, dh = layer.backward(dout[:, t, :])
            dhs_enc += dhs
            dhs_dec[:, t, :] = dh
        return dhs_enc, dhs_dec


class AttentionEncoder(Encoder):
    def forward(self, xs):
        xs = self.embed_layer.forward(xs)
        hs = self.lstm_layer.forward(xs)
        return hs

    def backward(self, dhs):
        dout = self.lstm_layer.backward(dhs)
        dout = self.embed_layer.backward(dout)
        return dout


class AttentionDecoder(BaseModel):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype("f")
        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype("f")
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype("f")
        lstm_b = np.zeros(4 * H).astype("f")
        affine_W = (rn(2 * H, V) / np.sqrt(2 * H)).astype("f")
        affine_b = np.zeros(V).astype("f")

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.attention = TimeAttention()
        self.affine = TimeAffine(affine_W, affine_b)
        layers = [self.embed, self.lstm, self.attention, self.affine]

        super().__init__(layers)

    def forward(self, xs, enc_hs):
        h = enc_hs[:, -1, :]
        self.lstm.set_state(h)

        out = self.embed.forward(xs)
        dec_hs = self.lstm.forward(out)
        c = self.attention.forward(enc_hs, dec_hs)
        out = np.concatenate((c, dec_hs), axis=2)
        score = self.affine.forward(out)
        return score

    def backward(self, dscore):
        dout = self.affine.backward(dscore)
        N, T, H2 = dout.shape
        H = H2 // 2

        dc, ddec_hs0 = dout[:, :, :H], dout[:, :, H:]
        denc_hs, ddec_hs1 = self.attention.backward(dc)
        ddec_hs = ddec_hs0 + ddec_hs1
        dout = self.lstm.backward(ddec_hs)
        dh = self.lstm.dh
        denc_hs[:, -1] += dh
        self.embed.backward(dout)

        return denc_hs

    def generate(self, enc_hs, start_id, sample_size):
        h = enc_hs[:, -1, :]
        self.lstm.set_state(h)

        word_ids = []
        char_id = start_id
        for _ in range(sample_size):
            x = np.array([[char_id]])
            out = self.embed.forward(x)
            dec_hs = self.lstm.forward(out)
            c = self.attention.forward(enc_hs, dec_hs)
            out = np.concatenate((c, dec_hs), axis=2)
            score = self.affine.forward(out)

            char_id = np.argmax(score.flatten())
            word_ids.append(char_id)

        return word_ids
