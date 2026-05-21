from nlp.seq2seq import Seq2seq
from nlp.attention_layer import AttentionEncoder, AttentionDecoder


class AttentionSeq2seq(Seq2seq):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        args = vocab_size, wordvec_size, hidden_size
        super().__init__(*args)

        self.encoder = AttentionEncoder(*args)
        self.decoder = AttentionDecoder(*args)

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads
