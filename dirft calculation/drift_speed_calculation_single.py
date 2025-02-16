import pandas as pd
import numpy as np
from pyproj import Transformer
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# 读取CSV文件
file_path = r"D:\OneDrive - 同济大学\无人机\Research\全移动场景的摄影测量\archived\北极浮标1.csv"  # 请替换为实际的CSV文件路径
data = pd.read_csv(file_path)

# 初始化投影转换器
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3995")

# 转换经纬度到投影坐标系
x, y = transformer.transform(data['Latitude'].values, data['Longitude'].values)

# 计算相邻观测点之间的时间差（以小时为单位）
data['DeviceDateTime'] = pd.to_datetime(data['DeviceDateTime'])
time_diff = (data['DeviceDateTime'].diff()).dt.total_seconds() / 3600

# 计算相邻观测点之间的距离（以米为单位）
dx = np.diff(x)
dy = np.diff(y)
distance = np.sqrt(dx**2 + dy**2)

# 计算速度（以千米/小时为单位）
speed = distance / time_diff[1:] / 1000

# 计算漂流方向（以度为单位，从正北方向顺时针测量）
direction = np.arctan2(dx, dy) * 180 / math.pi
direction = (direction + 360) % 360  # 将角度转换到0-360度范围

# 将结果添加到DataFrame中
data['Speed (km/h)'] = [np.nan] + speed.tolist()
data['Direction (degrees)'] = [np.nan] + direction.tolist()

# 绘制漂流轨迹图，并以不同颜色表示速度的大小
fig, ax = plt.subplots()
sc = ax.scatter(x[1:], y[1:], c=speed, cmap='viridis', norm=mcolors.Normalize(vmin=min(speed), vmax=max(speed)))
plt.colorbar(sc, label='Speed (km/h)')
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_title('Drift Trajectory with Speed Indication')
plt.show()


# 创建一个新的图形和子图
fig, ax1 = plt.subplots()

# 绘制速度的折线图
ax1.set_xlabel('Index')
ax1.set_ylabel('Speed (km/h)', color='tab:blue')
ax1.plot(data.index, data['Speed (km/h)'], color='tab:blue', label='Speed (km/h)')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# 创建第二个y轴用于绘制方向的折线图
ax2 = ax1.twinx()
ax2.set_ylabel('Direction (degrees)', color='tab:red')
ax2.plot(data.index, data['Direction (degrees)'], color='tab:red', label='Direction (degrees)')
ax2.tick_params(axis='y', labelcolor='tab:red')

# 添加标题
plt.title('Speed and Direction Over Time')

# 显示图例
fig.tight_layout()
fig.legend(loc='upper left', bbox_to_anchor=(0.1,0.9))

# 显示图形
plt.show()


# 打印结果
print(data)

# 保存结果到新的CSV文件
data.to_csv('output.csv', index=False)