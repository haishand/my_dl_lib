import numpy as np

from dataset import ptb

corpus, word_to_id, id_to_word = ptb.load_data("train")
print("corpus size:", len(corpus))
print("corpus[:30]:", corpus[:30])
print()
print("id_to_word[0]:", id_to_word[0])
print("id_to_word[1]:", id_to_word[1])
print("id_to_word[2]:", id_to_word[2])
print()
print("word_to_id['car']:", word_to_id["car"])
