"""
pandas分析数据
"""
import pandas as pd

train_df = pd.read_csv("./torch/california-house-prices/train.csv")
test_df = pd.read_csv("./torch/california-house-prices/test.csv")

print("训练集形状:", train_df.shape)
print("\n训练集字段:")
print(train_df.columns.tolist())
print("\n训练集前5行:")
print(train_df.head())

