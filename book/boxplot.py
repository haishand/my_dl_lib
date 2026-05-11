# 本案例使用箱线图展示海洋数据分布与异常值
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 解决中文显示
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 1. 准备海洋监测数据（含正常数据+少量异常值）
np.random.seed(88)
temp_data = np.random.normal(loc=22, scale=3, size=500)
# 手动插入10个异常值，模拟测量错误
outliers = np.random.uniform(low=5, high=10, size=10)
temp_data = np.concatenate([temp_data, outliers])

data = pd.DataFrame({"water_temp": temp_data})

# 2. 绘制箱线图
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=data,
    y="water_temp",  # 垂直箱线图
    color="#45b7d1",  # 海洋蓝色
    orient="v",  # v垂直 h水平
    width=0.6,
)

# 3. 设置图表信息
plt.title("海洋表层水温箱线图（异常值检测）", fontsize=14)
plt.ylabel("水温 (℃)", fontsize=12)
plt.grid(alpha=0.3)

# 保存并显示
plt.savefig("ocean_temp_boxplot.png", dpi=300, bbox_inches="tight")
plt.show()
