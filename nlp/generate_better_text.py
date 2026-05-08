import os, sys

sys.path.append(os.getcwd())
from nlp.better_rnnlm_gen import BetterRnnlmGen

sys.path.append(os.getcwd())

from dataset import ptb

corpus, word_to_id, id_to_word = ptb.load_data("train")
vocab_size = len(word_to_id)
corpus_size = len(corpus)

model = BetterRnnlmGen(vocab_size, 100, 100)
model.load_params()

start_words = ["the", "meaning", "of", "life", "is"]
start_ids = [word_to_id[w] for w in start_words]
skip_words = ["N", "<unk>", "$"]
skip_ids = [word_to_id[w] for w in skip_words]

word_ids = model.generate(start_ids, skip_ids, 10)
txt = " ".join([id_to_word[i] for i in word_ids])
txt = txt.replace("<eos>", "\n")
print(txt)
