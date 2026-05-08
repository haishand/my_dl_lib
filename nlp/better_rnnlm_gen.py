from common.xp import *
from common.functions import softmax
from nlp.better_rnnlm import BetterRnnLM


class BetterRnnlmGen(BetterRnnLM):
    def generate(self, start_ids, skip_ids=None, sample_size=100):
        word_ids = list(start_ids)

        x = word_ids[-1]
        while len(word_ids) < sample_size:
            x = np.array(x).reshape(1, -1)
            score = self.predict(x)
            p = softmax(score.flatten())

            sampled = np.random.choice(len(p), size=1, p=p)
            if (skip_ids is None) or (sampled not in skip_ids):
                x = sampled.item()
                word_ids.append(int(x))
        return word_ids
