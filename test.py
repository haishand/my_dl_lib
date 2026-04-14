import numpy as np

from common.util import *
import matplotlib.pyplot as plt

text = 'you say goodbye and i say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
word_matrix = create_co_matrix(corpus, vocab_size)

most_similar('you', word_to_id, id_to_word, word_matrix, top=5)

plt.annotate('you', xy=(0, 0), xytext=(10, 10), textcoords='offset points', arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
plt.show()