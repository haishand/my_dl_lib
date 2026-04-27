import os
import sys

sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import numpy as np
from common.optimizer import SGD
from dataset import ptb
from nlp.simple_rnnlm import SimpleRnnlm

# 设定超参数
batch_size = 10
wordvec_size = 100
hidden_size = 100
time_size = 5
lr = 0.1
max_epoch = 100

# 读入训练数据
corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus_size = 1000
corpus = corpus[:corpus_size]
vocab_size = int(max(corpus) + 1)

xs = corpus[:-1]
ts = corpus[1:]
data_size = len(xs)
print(f'corpus size: {corpus_size}, vocabulary size: {vocab_size}')

# 学习用的参数
max_iters = data_size // (batch_size * time_size)   # ?
time_idx = 0
total_loss = 0
loss_count = 0
ppl_list = []

# 生成模型
model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)

# 1. 计算读入mini-batch的各笔样本数据的开始位置
jump = (corpus_size - 1) // batch_size  # ?
offsets= [i * jump for i in range(batch_size)]

for epoch in range(max_epoch):
    for iter in range(max_iters):
        # 2. 获取mini-batch
        batch_x = np.empty((batch_size, time_size), dtype='i')
        batch_t = np.empty((batch_size, time_size), dtype='i')
        for t in range(time_size):
            for i, offset in enumerate(offsets):
                batch_x[i, t] = xs[(offset + time_idx) % data_size]
                batch_t[i, t] = ts[(offset + time_idx) % data_size]
            time_idx += 1
        
        # 计算梯度，更新参数
        loss = model.forward(batch_x, batch_t)
        model.backward()
        optimizer.update(model.params, model.grads)
        total_loss += loss
        loss_count += 1

    # 3. 对各个epoch的ppl进行评价
    ppl = np.exp(total_loss/loss_count)
    print(f'|epcoh {epoch} | ppl {ppl}.2f')
    ppl_list.append(float(ppl))
    total_loss, loss_count = 0, 0
        
x = np.arange(len(ppl_list))
plt.plot(x, ppl_list)
plt.xlabel('epoch')
plt.ylabel('ppl')
plt.show()