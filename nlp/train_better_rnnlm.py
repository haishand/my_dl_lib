import os, sys

sys.path.append(os.getcwd())
from common.util import eval_perplexity

from common.trainer import RnnlmTrainer
from nlp.better_rnnlm import BetterRnnLM
from common.optimizer import SGD
from dataset import ptb

# 设定超参
max_epoch = 4
batch_size = 20
wordvec_size = 650  # 词向量维度
hidden_size = 650  # 隐藏层维度
dropout_ratio = 0.5  # dropout比例
lr = 20.0  # 学习率
max_grad = 0.25  # 用于梯度裁剪的阈值
time_size = 35  # RNN展开的时间步数

# 加载数据
corpus, word_to_id, id_to_word = ptb.load_data("train")
corpus_val, _, _ = ptb.load_data("val")
corpus_test, _, _ = ptb.load_data("test")

vocab_size = len(word_to_id)  # 词汇表大小
xs = corpus[:-1]
ts = corpus[1:]

model = BetterRnnLM(vocab_size, wordvec_size, hidden_size, dropout_ratio)
optimizer = SGD(lr)
trainer = RnnlmTrainer(model, optimizer)

best_ppl = float("inf")
for epoch in range(max_epoch):
    trainer.fit(
        xs,
        ts,
        max_epoch=1,
        batch_size=batch_size,
        time_size=time_size,
        max_grad=max_grad,
    )

    model.reset_state()
    ppl = eval_perplexity(model, corpus_val)  # 训练数据上的困惑度
    print("valid perplexity: ", ppl)

    if best_ppl > ppl:
        best_ppl = ppl
        model.save_params()
    else:
        lr /= 4.0
        optimizer.lr = lr
    model.reset_state()
    print("-" * 50)
