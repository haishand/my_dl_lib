import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 1. 构造简单语料 & 词映射
text = "i love deep learning i love pytorch i love rnn language generation"
words = text.split()
vocab = sorted(list(set(words)))
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for i, w in enumerate(vocab)}
vocab_size = len(vocab)

# 超参
embed_dim = 16
hidden_dim = 32
seq_len = 3
lr = 0.001
epochs = 800


# 2. 构建时序训练样本
def build_sample(text, word2idx, seq_len):
    words = text.split()
    xs, ys = [], []
    for i in range(len(words) - seq_len):
        x = [word2idx[w] for w in words[i : i + seq_len]]
        y = word2idx[words[i + seq_len]]
        xs.append(x)
        ys.append(y)
    return torch.LongTensor(xs), torch.LongTensor(ys)


x_data, y_data = build_sample(text, word2idx, seq_len)


# 3. 简易RNN语言模型
class RNNLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.embedding(x)
        out, hidden = self.lstm(emb, hidden)
        out = self.fc(out[:, -1, :])  # 取最后时序输出预测
        return out, hidden


# 4. 初始化模型、损失、优化器
model = RNNLM(vocab_size, embed_dim, hidden_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# 5. 训练
for epoch in range(epochs):
    optimizer.zero_grad()
    pred, _ = model(x_data)
    loss = criterion(pred, y_data)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")


# 6. 文本生成函数
def generate(start_seq, gen_len=10):
    model.eval()
    current = [word2idx[w] for w in start_seq]
    gen_words = start_seq.copy()
    hidden = None
    with torch.no_grad():
        for _ in range(gen_len):
            x = torch.LongTensor([current])
            logits, hidden = model(x, hidden)
            next_idx = torch.argmax(logits, dim=-1).item()
            gen_words.append(idx2word[next_idx])
            current = current[1:] + [next_idx]
    return " ".join(gen_words)


# 测试生成
start = ["i", "love"]
result = generate(start, gen_len=8)
print("\n生成文本：", result)
