import os
import glob
import pandas as pd
import numpy as np
from pyproj import Proj, Transformer

def load_and_preprocess_data(folder_path):
    files = glob.glob(os.path.join(folder_path, "*.csv"))
    if len(files) != 4:
        raise ValueError("文件夹必须包含4个CSV文件")

    dfs = []
    for file in files:
        df = pd.read_csv(file, parse_dates=['datetime'])
        df = df.sort_values('datetime').drop_duplicates('datetime')
        df = df[['datetime', 'longitude', 'latitude']]
        dfs.append(df.set_index('datetime'))
    return dfs

def calculate_common_timeline(dfs, interval):
    start_time = max(df.index.min() for df in dfs)
    end_time = min(df.index.max() for df in dfs)
    time_grid = pd.date_range(start_time, end_time, freq=interval)
    return time_grid

def convert_to_polar_coordinates(dfs, time_grid):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3413")
    
    polar_dfs = []
    for df in dfs:
        df_resampled = df.reindex(time_grid).interpolate(method='time')
        positions = []
        for _, row in df_resampled.iterrows():
            x, y = transformer.transform(row['latitude'], row['longitude'])
            positions.append((x, y))
        polar_df = pd.DataFrame(positions, index=time_grid, columns=['x', 'y'])
        polar_dfs.append(polar_df)
    return polar_dfs

def calculate_rotation_angle(current, next_pos):
    """计算二维旋转角度（弧度）"""
    H = current.T @ next_pos
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # 确保是纯旋转（无反射）
    if np.linalg.det(R) < 0:
        Vt[-1,:] *= -1
        R = Vt.T @ U.T
    return np.arctan2(R[1,0], R[0,0])

def calculate_parameters(polar_dfs, time_grid, interval):
    results = []
    delta_t = pd.to_timedelta(interval).total_seconds()
    
    for i in range(len(time_grid)-1):
        current_time = time_grid[i]
        next_time = time_grid[i+1]
        
        # 添加维度检查
        current_positions = np.array([np.ravel(df.loc[current_time].values) for df in polar_dfs])
        next_positions = np.array([np.ravel(df.loc[next_time].values) for df in polar_dfs])
        
        # 强制展平维度
        current_centroid = np.mean(current_positions, axis=0).flatten()
        next_centroid = np.mean(next_positions, axis=0).flatten()
        
        displacement = next_centroid - current_centroid
        displacement = displacement.flatten()  # 确保一维        
        speed = np.linalg.norm(displacement) / delta_t
        
        # 修正参数顺序
        direction = np.degrees(np.arctan2(displacement[1], displacement[0])) % 360
        
        # 计算相对位置的旋转角度
        current_relative = current_positions - current_centroid
        next_relative = next_positions - next_centroid
        
        # 计算刚体旋转角度（二维）
        rotation_angle = calculate_rotation_angle(current_relative, next_relative)
        angular_speed = np.abs(np.degrees(rotation_angle)) / delta_t  # 度/秒（取绝对值）
        
        results.append({
            'start_time': current_time,
            'end_time': next_time,
            'speed_mps': speed,
            'direction_deg': direction,
            'angular_speed_degps': angular_speed
        })
    
    return pd.DataFrame(results)

def main():
    folder_path = r"D:\Working_Project\Arctic_2024_Shuhang\Data\冰上GNSS控制点数据\0830_冰基长基线浮标GNSS_O文件转换\解算数据"
    interval = '1h'   
    
    dfs = load_and_preprocess_data(folder_path)
    time_grid = calculate_common_timeline(dfs, interval)
    polar_dfs = convert_to_polar_coordinates(dfs, time_grid)
    results = calculate_parameters(polar_dfs, time_grid, interval)
    
    print("\n计算结果：")
    print(results.to_string(index=False))
    results.to_csv('polar_results.csv', index=False)
    print("\n结果已保存到 polar_results.csv")

if __name__ == "__main__":
    main()