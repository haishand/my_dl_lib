import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===================== 1. 读取数据 =====================
train_df = pd.read_csv("./torch/california-house-prices/train.csv")
test_df = pd.read_csv("./torch/california-house-prices/test.csv")

print("=" * 50)
print("训练集形状:", train_df.shape)
print("测试集形状:", test_df.shape)
print("=" * 50)

# ===================== 2. 查看前5行数据 =====================
print("\n【训练集前5行】")
print(train_df.head())

# ===================== 3. 查看列名 =====================
print("\n【所有列名】")
print(train_df.columns.tolist())

# ===================== 4. 数据类型 & 基本信息 =====================
print("\n【数据类型信息】")
train_df.info()

# ===================== 5. 统计描述（均值、最大最小、分位数） =====================
print("\n【数据统计描述】")
print(train_df.describe())

# ===================== 6. 检查缺失值（非常重要） =====================
print("\n【训练集缺失值统计】")
missing_train = train_df.isnull().sum().sort_values(ascending=False)
print(missing_train[missing_train > 0])

print("\n【测试集缺失值统计】")
missing_test = test_df.isnull().sum().sort_values(ascending=False)
print(missing_test[missing_test > 0])

# ===================== 7. 查看标签列（房价）分布 =====================
target_col = train_df["Sold Price"].name
print(f"\n【标签列：{target_col}】")
print(train_df[target_col].describe())

# ===================== 8. 特征相关性分析（和房价的关系） =====================
print("\n【特征与房价相关性（从高到低）】")
train_numeric_df = train_df.select_dtypes(include=[np.number])
test_numeric_df = test_df.select_dtypes(include=[np.number])
corr = train_numeric_df.corr()[target_col].sort_values(ascending=False)
print(corr)

# ===================== 9. 异常值检测（3σ原则） =====================
print("\n【异常值检测】")


def detect_outliers(df, col):
    mean = df[col].mean()
    std = df[col].std()
    lower = mean - 3 * std
    upper = mean + 3 * std
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    return len(outliers)


for col in train_df.select_dtypes(include=[np.number]).columns:
    if col != target_col:
        n = detect_outliers(train_df, col)
        if n > 0:
            print(f"{col} 异常值数量: {n}")

# ===================== 10. 训练集 vs 测试集分布对比 =====================
print("\n【训练集 / 测试集 特征均值对比】")
train_mean = train_numeric_df.mean()
test_mean = test_numeric_df.mean()
compare = pd.DataFrame({"训练集均值": train_mean, "测试集均值": test_mean})
print(compare.head(10))
