# 海洋水温与盐度回归拟合曲线图
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 设置中文、负号正常显示
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 1. 构造模拟海洋数据：水温 & 盐度
np.random.seed(66)
n = 500
temp = np.random.normal(loc=22, scale=2.5, size=n)
# 构造负相关：温度越高，盐度略低，加噪声
salinity = 35 - 0.25 * temp + np.random.normal(0, 0.8, size=n)

df = pd.DataFrame({"水温": temp, "盐度": salinity})

# 2. 创建画布、绘制回归曲线
plt.figure(figsize=(9, 6))

sns.regplot(
    data=df,
    x="水温",
    y="盐度",
    color="#1f77b4",
    # 散点样式
    scatter_kws={"s": 25, "alpha": 0.6},
    # 回归直线样式
    line_kws={"lw": 2, "linestyle": "-"},
    ci=95,  # 显示95%置信区间阴影
)

# 3. 图表美化
plt.title("海洋水温与盐度回归拟合曲线图", fontsize=14)
plt.xlabel("海水温度 (℃)", fontsize=12)
plt.ylabel("海水盐度 (PSU)", fontsize=12)
plt.grid(alpha=0.3)

# 保存图片
plt.savefig("ocean_regplot.png", dpi=300, bbox_inches="tight")
plt.show()
