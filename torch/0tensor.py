import torch
import numpy as np

print("===== 1. 创建 Tensor =====")
# 1. 从列表创建
a = torch.tensor([[1, 2, 3], [4, 5, 6]])
print("从列表创建：\n", a)

# 2. 全 0 / 全 1 / 随机
b = torch.zeros(2, 3)   # 2行3列全0
c = torch.ones(2, 3)    # 2行3列全1
d = torch.rand(2, 3)    # 0~1随机数
print("\n全0：\n", b)
print("全1：\n", c)
print("随机：\n", d)

# 3. 和已有 tensor 同形状
e = torch.ones_like(a)
print("\n和 a 同形状全1：\n", e)

print("\n===== 2. Tensor 属性（形状、数据类型、设备） =====")
print("形状 shape:", a.shape)    # 最常用
print("数据类型 dtype:", a.dtype)
print("设备 device:", a.device)
print("维度数 dim:", a.dim())

print("\n===== 3. 索引与切片（重点！） =====")
# PyTorch 和 NumPy 完全一样：[行, 列]
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

print("取单个元素 x[1,2]:", x[1, 2].item())  # .item() 转普通数字
print("取前2行，前2列 x[:2, :2]:\n", x[:2, :2])
print("取第1列所有行 x[:, 1]:\n", x[:, 1])

print("\n===== 4. 数学运算 =====")
m = torch.tensor([[1, 2], [3, 4]])
n = torch.tensor([[5, 6], [7, 8]])

# 加减乘除（对应元素运算）
print("m + n:\n", m + n)
print("m * n:\n", m * n)

# 矩阵乘法（深度学习核心）
print("矩阵乘法 matmul:\n", torch.matmul(m, n))

print("\n===== 5. 形状变换 =====")
t = torch.rand(2, 3)
print("原形状 (2,3):\n", t)

# 展平
print("展平 flatten:", t.flatten())

# 重塑 reshape
t_reshaped = t.reshape(3, 2)
print("reshape (3,2):\n", t_reshaped)

print("\n===== 6. Tensor ↔ NumPy 互转 =====")
# Tensor → NumPy
np_arr = a.numpy()
print("Tensor 转 NumPy:\n", np_arr)

# NumPy → Tensor
np_data = np.array([1, 2, 3])
tensor_from_np = torch.from_numpy(np_data)
print("NumPy 转 Tensor:\n", tensor_from_np)

print("\n===== 7. 设备切换（CPU / GPU） =====")
cpu_tensor = torch.tensor([1, 2, 3])
print("CPU Tensor:", cpu_tensor.device)

# 如果有 GPU 就迁移
if torch.cuda.is_available():
    gpu_tensor = cpu_tensor.cuda()  # 或者 to("cuda")
    print("GPU Tensor:", gpu_tensor.device)
else:
    print("未检测到 GPU，保持 CPU 运行")

print("\n===== 8. 常用统计函数 ====="
st = torch.tensor([1.0, 2.0, 3.0, 4.0])
print("和 sum:", st.sum().item())
print("平均值 mean:", st.mean().item())
print("最大值 max:", st.max().item())