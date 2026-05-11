# 本案例使用热力图展示海洋环境多指标相关性
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 解决中文显示
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 1. 生成模拟海洋环境监测数据
np.random.seed(88)
data = pd.DataFrame(
    {
        "水温": np.random.normal(22, 3, 500),
        "盐度": np.random.normal(34, 1.2, 500),
        "深度": np.random.exponential(20, 500),
        "pH值": np.random.normal(8.1, 0.15, 500),
    }
)

# 2. 计算相关矩阵
corr_matrix = data.corr()

# 3. 绘制相关性热力图
plt.figure(figsize=(8, 6))

sns.heatmap(
    corr_matrix,  # 相关矩阵
    cmap="coolwarm",  # 蓝-红配色（适合表示正负相关）
    annot=True,  # 显示相关系数数值
    fmt=".2f",  # 数值保留2位小数
    linewidths=0.5,  # 格子间线条宽度
    cbar=True,  # 显示右侧颜色条
    vmin=-1,
    vmax=1,  # 颜色范围固定在 [-1,1]
)

# 4. 设置图表标题
plt.title("海洋环境指标相关性热力图", fontsize=14)

# 保存并显示图片
plt.savefig("ocean_corr_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()
