from drift_utils import *
import numpy as np
import pandas as pd
import ruptures as rpt
from scipy.stats import mode
import matplotlib.pyplot as plt



#参数	影响	典型值范围	调整策略
#initial_interval	基础计算分辨率	1T-30T	应小于预期最小分辨率
#penalty_value	检测灵敏度	10-1000	根据速度波动幅度调整，值越大分段越少
# model	变点检测类型	'l1'/'l2'/'rbf'	对速度突变用'l2'，渐变用'rbf'

initial_interval = '10s'
penalty_value = 10


def detect_optimal_interval(folder_path, initial_interval='10s', penalty_value=10):
    """
    确定最优时间分辨率的完整流程
    :param folder_path: 数据文件夹路径
    :param initial_interval: 初始时间间隔（用于参数计算）
    :param penalty_value: 变点检测惩罚系数（越大分段越少）
    :return: 推荐分辨率（秒）、变点信息、间隔统计
    """
    # 步骤1：计算基础参数
    dfs = load_and_preprocess_data(folder_path)
    time_grid = calculate_common_timeline(dfs, initial_interval)
    polar_dfs = convert_to_polar_coordinates(dfs, time_grid)
    param_df = calculate_parameters(polar_dfs, time_grid, initial_interval)
    
    # 步骤2：准备速度时间序列
    speed_series = param_df['speed_mps'].values
    timestamps = param_df['start_time'].values
    
    # 步骤3：PELT变点检测
    algo = rpt.Pelt(model="rbf").fit(speed_series)
    change_points = algo.predict(pen=penalty_value)
    
    # 处理变点索引（排除最后一个点）
    valid_cp = [cp for cp in change_points if cp < len(speed_series)]
    
    if len(valid_cp) < 2:
        print("警告：未检测到显著变点，建议降低penalty_value")
        return None, None, None
    
    # 步骤4：计算间隔统计
    cp_times = timestamps[valid_cp]
    intervals = np.diff(cp_times).astype('timedelta64[s]').astype(int)
    
    # 找到众数间隔
    mode_interval = mode(intervals).mode
    
    # 转换为人类可读格式
    def seconds_to_resolution(seconds):
        units = [(604800, 'W'), (86400, 'D'), (3600, 'H'), (60, 'T')]
        for divisor, unit in units:
            if seconds % divisor == 0:
                return f"{seconds//divisor}{unit}"
        return f"{seconds}S"
    
    recommended_res = seconds_to_resolution(mode_interval)
    
    # 步骤5：整理输出结果
    cp_info = []
    for i, cp in enumerate(valid_cp[:-1]):
        cp_info.append({
            'segment_id': i+1,
            'start': timestamps[cp],
            'end': timestamps[valid_cp[i+1]],
            'duration_sec': intervals[i],
            'mean_speed': np.mean(speed_series[cp:valid_cp[i+1]])
        })
    
    # 修改后的间隔统计部分
    interval_stats = pd.DataFrame({
        'total_segments': [len(intervals) + 1],
        'mean_interval_seconds': [np.mean(intervals)],
        'median_interval_seconds': [np.median(intervals)],
        'mode_interval_seconds': [mode_interval],
        'min_interval': [np.min(intervals)],
        'max_interval': [np.max(intervals)],
        'std_deviation': [np.std(intervals)]
    })

    # 可选：单独保存原始间隔数据
    interval_details = pd.DataFrame({
        'interval_seconds': intervals,
        'interval_index': range(1, len(intervals)+1)
    })
    plot_speed_changes(param_df, valid_cp)

    return recommended_res, cp_info, interval_stats, interval_details


def plot_speed_changes(param_df, change_points):
    plt.figure(figsize=(12, 6))
    
    # 绘制速度曲线
    plt.plot(param_df['start_time'], param_df['speed_mps'], 
             label='Speed (m/s)', alpha=0.6)
    
    # 标记变点
    cp_times = param_df['start_time'].iloc[change_points[:-1]]
    for cp in cp_times:
        plt.axvline(x=cp, color='red', linestyle='--', alpha=0.5)
    
    plt.title('Speed Profile with Change Points')
    plt.xlabel('Time')
    plt.ylabel('Speed (m/s)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('speed_changes.png')
    plt.show()

# 在detect_optimal_interval函数末尾添加：

# 使用示例
if __name__ == "__main__":
    folder = r"D:\Working_Project\Arctic_2024_Shuhang\Data\冰上GNSS控制点数据\0830_冰基长基线浮标GNSS_O文件转换\解算数据"
    resolution, change_points, stats, details = detect_optimal_interval(folder,initial_interval,penalty_value)
    
    print(f"\n推荐时间分辨率：{resolution}")
    print("\n变点统计：")
    print(pd.DataFrame(change_points).to_string(index=False))
    print("\n间隔统计：")
    print(stats.to_string(index=False))
    print(details.head(3).to_string(index=False))

