import os
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from matplotlib.collections import LineCollection

SPEED_MAX = .5

def plot_arctic_trajectories(folder_path, step=1, output_file='trajectory_plot.png'):
    # 初始化参数
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    proj = ccrs.NorthPolarStereo(central_longitude=-45)
    speed_min = np.inf
    speed_max = -np.inf
    lons, lats = [], []

    # 第一次遍历：收集数据范围信息
    for file in all_files:
        df = pd.read_csv(os.path.join(folder_path, file)).iloc[::step]
        if len(df) < 2:
            continue
        speed_min = min(speed_min, df['speed'].min())
        speed_max = max(speed_max, df['speed'].max())
        lons.extend(df['longitude'])
        lats.extend(df['latitude'])

    # 设置地图范围和投影
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_extent([min(lons), max(lons), min(lats), max(lats)], 
                 crs=ccrs.PlateCarree())
    
    # 添加地理特征
    ax.add_feature(cfeature.LAND.with_scale('50m'), zorder=1)
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), zorder=2)
    ax.gridlines(linestyle=':', color='gray')

    # 创建颜色标准化对象
    norm = plt.Normalize(speed_min, SPEED_MAX)
    cmap = plt.get_cmap('viridis')

    # 第二次遍历：绘制轨迹
    for file in all_files:
        df = pd.read_csv(os.path.join(folder_path, file)).iloc[::step]
        if len(df) < 2:
            continue
        
        # 转换坐标到投影坐标系
        points = proj.transform_points(ccrs.PlateCarree(), 
                                      df['longitude'].values, 
                                      df['latitude'].values)
        x, y = points[:, 0], points[:, 1]
        
        # 创建线段集合
        segments = np.array([[[x[i], y[i]], [x[i+1], y[i+1]]] 
                           for i in range(len(x)-1)])
        speeds = df['speed'].values[:-1]
        
        lc = LineCollection(segments, cmap=cmap, norm=norm,
                          linewidth=1.5, zorder=3)
        lc.set_array(speeds)
        ax.add_collection(lc)

    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.05)
    cbar.set_label('Speed (m/s)', fontsize=12)

    # 保存或显示结果
    plt.title('Arctic Buoy Trajectories with Speed Coloring', fontsize=14)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()

# 使用示例
folder = r"D:\Working_Project\Arctic_2024_Shuhang\Data\冰上GNSS控制点数据\0830_冰基长基线浮标GNSS_O文件转换\解算数据"
# folder = r"D:\Work\Project data\Arctic_2024\Data\冰上GNSS控制点数据\20240904长期冰站GNSS观测数据\20240904长期冰站定位结果\248"

plot_arctic_trajectories(folder, step=60)