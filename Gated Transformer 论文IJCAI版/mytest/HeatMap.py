import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
flights_long = sns.load_dataset("flights")
flights = flights_long.pivot("month", "year", "passengers")
# 绘制x-y-z的热力图，比如 年-月-销量 的热力图
f, ax = plt.subplots(figsize=(9, 6))
#绘制热力图，还要将数值写到热力图上
sns.heatmap(flights, annot=True, fmt="d", ax=ax)
#设置坐标字体方向
label_y = ax.get_yticklabels()
plt.setp(label_y, rotation=360, horizontalalignment='right')
label_x = ax.get_xticklabels()
plt.setp(label_x, rotation=45, horizontalalignment='right')
plt.show()