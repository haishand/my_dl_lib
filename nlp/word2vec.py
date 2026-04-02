import os, sys

sys.path.append(os.getcwd())
import numpy as np
from common.util import convert_one_hot, preprocess, create_contexts_target
from common.optimizer import SGD
from CBOW import SimpleCBOW
from common.trainer import Trainer

max_epoch = 1000
batch_size = 3
window_size = 1
hidden_size = 5

text = "You say goodbye and I say hello."
corpus, word_to_id, id_to_word = preprocess(text)

contexts, target = create_contexts_target(corpus, window_size)

vocab_size = len(word_to_id)
target = convert_one_hot(target, vocab_size)
contexts = convert_one_hot(contexts, vocab_size)


optimizer = SGD(lr=0.1)
model = SimpleCBOW(vocab_size, hidden_size)
trainer = Trainer(model, optimizer)
trainer.fit(contexts, target, max_epoch, batch_size)
# trainer.plot()

print(contexts[0])
my_contexts = np.array([[[1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0]]])
pred = model.predict(my_contexts)
print(id_to_word[np.argmax(pred)])

word_vecs = model.word_vecs
for word_id, word in id_to_word.items():
    print(word, word_vecs[word_id])
