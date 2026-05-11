# 本案例使用直方图+密度曲线展示海洋表层水温分布特征
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 解决中文显示问题
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 1. 准备数据（模拟500个海洋表层水温数据）
np.random.seed(88)
data = pd.DataFrame({"water_temp": np.random.normal(loc=21.5, scale=2.8, size=500)})

# 2. 绘制直方图+密度曲线
plt.figure(figsize=(10, 6))

sns.histplot(
    data=data,
    x="water_temp",
    kde=True,  # 开启核密度曲线
    bins=28,  # 直方图柱子数量
    color="#3498db",  # 海洋蓝色
    alpha=0.7,  # 透明度
)

# 3. 设置图表信息
plt.title("海洋表层水温分布直方图+密度曲线", fontsize=14)
plt.xlabel("水温 (℃)", fontsize=12)
plt.ylabel("观测频次 / 密度", fontsize=12)
plt.grid(alpha=0.3)

# 保存并显示图片
plt.savefig("water_temp_dist.png", dpi=300, bbox_inches="tight")
plt.show()
