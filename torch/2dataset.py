import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


# ======================
# 1. 自定义 Dataset（必须继承 Dataset）
# ======================
class MyDataset(Dataset):
    # 初始化：把数据读进来
    def __init__(self):
        # 模拟数据：10个样本，每个样本4个特征
        self.data = torch.tensor(np.random.randn(10, 4), dtype=torch.float32)
        # 模拟标签：10个标签
        self.label = torch.tensor(np.random.randint(0, 2, size=(10,)), dtype=torch.long)

    # 返回总共有多少条数据
    def __len__(self):
        return len(self.data)

    # 返回第 index 条数据（核心！）
    def __getitem__(self, index):
        x = self.data[index]
        y = self.label[index]
        return x, y


# ======================
# 2. 创建数据集
# ======================
dataset = MyDataset()

# 测试取一条数据
x_sample, y_sample = dataset[0]
print("单条数据形状:", x_sample.shape)
print("单条标签:", y_sample)
print("总数据量:", len(dataset))

# ======================
# 3. 创建 DataLoader（批量加载）
# ======================
dataloader = DataLoader(
    dataset,
    batch_size=3,  # 每批3个数据
    shuffle=True,  # 打乱数据
    num_workers=0,  # 单线程（Windows写0，Linux可写2）
)

# ======================
# 4. 用 for 循环遍历加载（训练时就这样写）
# ======================
print("\n===== 开始批量加载数据 =====")
for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
    print(f"第 {batch_idx} 批")
    print("数据形状:", x_batch.shape)  # torch.Size([3,4]) → batch=3, feature=4
    print("标签形状:", y_batch.shape)
    print("---")
