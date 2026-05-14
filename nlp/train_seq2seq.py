import sys, os

sys.path.append(os.getcwd())
from dataset import sequence
from seq2seq import Seq2seq
from common.optimizer import Adam
from common.trainer import Trainer
from common.util import eval_seq2seq

# 读入数据
(x_train, t_train), (x_test, t_test) = sequence.load_data("addition.txt")
char_to_id, id_to_char = sequence.get_vocab()

reversed = False
if reversed:
    x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]

# 设定超参
vocab_size = len(char_to_id)
wordvec_size = 16
hidden_size = 128
batch_size = 128
max_epoch = 10
max_grad = 5.0

# 生成模型
model = Seq2seq(vocab_size, wordvec_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

acc_list = []
for epoch in range(max_epoch):
    trainer.fit(x_train, t_train, max_epoch=1, batch_size=batch_size, max_grad=max_grad)
    correct_num = 0
    for i in range(len(x_test)):
        question, correct = x_test[[i]], t_test[[i]]
        verbose = i < 10
        correct_num += eval_seq2seq(
            model, question, correct, id_to_char, verbos=True, is_reverse=reversed
        )
    acc = float(correct_num) / len(x_test)
    acc_list.append(acc)
    print("accuracy: %.3f%%" % (acc * 100))
