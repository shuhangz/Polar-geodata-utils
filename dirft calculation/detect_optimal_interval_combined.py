from drift_utils import *
import numpy as np
import pandas as pd
import ruptures as rpt
from scipy.stats import mode
import matplotlib.pyplot as plt

# 参数	调整方法	典型值范围
# initial_interval	应小于预期最小分辨率	1T-10T
# penalty	值越大分段越少	速度：10-100
# 角速度：1-10（因角速度变化更敏感）

initial_interval = '10s'
penalty_value_velocity = 60
penalty_value_angular = 30

def detect_single_param_interval(param_series, timestamps, penalty=50):
    """改进版单参数检测，增加边界保护"""
    if len(param_series) < 2:
        return None  # 数据不足
    
    try:
        # 执行PELT检测
        algo = rpt.Pelt(model="rbf").fit(param_series)
        raw_change_points = algo.predict(pen=penalty)
        
        # 严格过滤无效变点
        valid_cp = [cp for cp in raw_change_points 
                   if 0 < cp < len(param_series)]  # 排除首尾端点
        valid_cp = sorted(list(set(valid_cp)))  # 去重排序
        
        # 二次验证：确保时间戳存在
        existing_indices = []
        for cp in valid_cp:
            try:
                _ = timestamps[cp]  # 检查索引有效性
                existing_indices.append(cp)
            except IndexError:
                continue
                
        if len(existing_indices) < 1:
            return None
        
        # 计算间隔
        cp_times = timestamps[existing_indices]
        intervals = np.diff(cp_times).astype('timedelta64[s]').astype(int)
        
        return intervals, existing_indices
        
    except Exception as e:
        print(f"变点检测异常：{str(e)}")
        return None

def get_recommended_interval(intervals, fallback=3600):
    """带降级策略的间隔推荐"""
    if intervals is None or len(intervals) == 0:
        return fallback  # 默认1小时
    
    try:
        mode_val = mode(intervals).mode
        # 合理性检查：间隔不应超过24小时
        return min(mode_val, 86400)  
    except:
        # 计算失败时返回中位数
        return np.median(intervals).astype(int)

def simple_interval_detection(folder_path, initial_interval='10T'):
    # 数据加载（复用原有函数）
    dfs = load_and_preprocess_data(folder_path)
    time_grid = calculate_common_timeline(dfs, initial_interval)
    polar_dfs = convert_to_polar_coordinates(dfs, time_grid)
    param_df = calculate_parameters(polar_dfs, time_grid, initial_interval)
    
    # 分别检测两个参数
    speed_intervals,valid_cp_speed = detect_single_param_interval(
        param_df['speed_mps'].values, 
        param_df['start_time'].values,
        penalty_value_velocity
    )
    angular_intervals, valid_cp_angular = detect_single_param_interval(
        param_df['angular_speed_degps'].values,
        param_df['start_time'].values,
        penalty_value_angular
    )
    
    # 获取推荐值
    speed_rec = get_recommended_interval(speed_intervals)
    angular_rec = get_recommended_interval(angular_intervals)
    
    # 处理未检测到变点的情况
    recommendations = []
    if speed_rec: recommendations.append(speed_rec)
    if angular_rec: recommendations.append(angular_rec)
    
    # 取最小值，如果都无效则返回None
    final_rec = min(recommendations) if recommendations else None
    
    
    # 转换为可读格式
    def sec_to_readable(seconds):
        for unit in ['W', 'D', 'H']:
            if seconds % (7*86400) == 0 and unit == 'W':
                return f"{int(seconds/(7*86400))}{unit}"
            if seconds % 86400 == 0 and unit == 'D':
                return f"{int(seconds/86400)}{unit}"
            if seconds % 3600 == 0 and unit == 'H':
                return f"{int(seconds/3600)}{unit}"
        return f"{seconds}S"
    
    plot_dual_detection(param_df,valid_cp_speed,valid_cp_angular)

    return {
        'speed_interval': sec_to_readable(speed_rec) if speed_rec else None,
        'angular_interval': sec_to_readable(angular_rec) if angular_rec else None,
        'final_recommendation': sec_to_readable(final_rec) if final_rec else "Not detected"
    }

def plot_dual_detection(param_df, speed_cp, angular_cp):
    plt.figure(figsize=(12,8))
    
    # 速度子图
    plt.subplot(211)
    plt.plot(param_df['start_time'], param_df['speed_mps'], label='Speed')
    for cp in speed_cp:
        plt.axvline(x=param_df['start_time'].iloc[cp], color='blue', linestyle='--', alpha=0.5)
    plt.legend()
    
    # 角速度子图
    plt.subplot(212)
    plt.plot(param_df['start_time'], param_df['angular_speed_degps'], color='orange', label='Angular Speed')
    for cp in angular_cp:
        plt.axvline(x=param_df['start_time'].iloc[cp], color='red', linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('dual_detection.png')


# 使用示例
if __name__ == "__main__":
    folder = r"D:\Working_Project\Arctic_2024_Shuhang\Data\冰上GNSS控制点数据\0830_冰基长基线浮标GNSS_O文件转换\解算数据"
    result = simple_interval_detection(folder,initial_interval)
    print(f"""
    速度推荐间隔: {result['speed_interval']}
    角速度推荐间隔: {result['angular_interval']}
    最终推荐值: {result['final_recommendation']}
    """)
