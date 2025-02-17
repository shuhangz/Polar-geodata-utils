from drift_utils import *
import numpy as np
import pandas as pd
import ruptures as rpt
from scipy.stats import mode
import matplotlib.pyplot as plt

# 用户希望程序在输出推荐的时间间隔后，还能输出对应的误差。现在我要考虑如何实现这个需求。首先，我需要明确什么是用户所指的“误差”。可能是指推荐的时间间隔与实际数据变化之间的误差，或者是指不同参数推荐值之间的差异？

# 根据之前的对话，用户提到的误差可能是指在每个推荐的间隔内，速度和角速度的平均变化量或波动情况。例如，如果推荐的时间间隔是30分钟，那么在这30分钟内，速度和角速度的变化幅度是多少，这个可以作为误差的衡量标准。

# 接下来，我需要确定如何计算这个误差。对于每个检测到的时间间隔段，可以计算该段内参数的均方误差（MSE）或平均绝对误差（MAE），反映该段内的波动情况。然后将所有段的误差进行平均或加权平均，作为整体的误差指标。

# 另外，用户可能需要知道推荐间隔的可靠性，比如如果误差很大，说明这个间隔可能不够精确，反之则更可靠。因此，误差指标应该能够反映推荐间隔的稳定性。

# 还需要考虑如何将误差值整合到现有的代码结构中。现有的代码在检测变点后，会计算每个分段的时间间隔，并推荐众数作为最终结果。现在需要在计算间隔的同时，计算每个分段的误差，并最终汇总输出。

# 可能的实现步骤：

# 1. 在检测每个参数的变点后，遍历每个分段，计算该分段内参数的均方误差或平均绝对误差。
# 2. 将所有分段的误差进行平均，得到该参数的整体误差。
# 3. 在最终推荐时，输出速度和角速度各自的误差，以及综合误差（如两者的平均值或最大值）。
# 4. 修改现有的返回结构，加入误差信息，并确保在输出时展示这些误差值。

# 需要注意的是，当某个参数没有检测到变点时，如何处理误差。可能需要设置默认值或提示用户数据不足。

# 此外，用户可能需要误差的单位和解释，例如速度的误差是m/s²，角速度的误差是deg/s²，或者是相对值如百分比。需要明确误差的计算方式和单位，以便用户理解。


# 参数	调整方法	典型值范围
# initial_interval	应小于预期最小分辨率	1T-10T
# penalty	值越大分段越少	速度：10-100
# 角速度：1-10（因角速度变化更敏感）

# ### 误差指标说明

# | 误差类型 | 计算方法 | 物理意义 |
# |---------|----------|----------|
# | 分段均方误差 (MSE) | $\frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2$ | 表征每个稳定段内参数的波动性 |
# | 综合误差 | 众数间隔对应分段的平均MSE | 反映推荐间隔的稳定性 |

initial_interval = '10s'
penalty_value_velocity = 60
penalty_value_angular = 30

def calculate_segment_errors(param_series, change_points):
    """计算每个分段的均方误差"""
    errors = []
    prev = 0
    for cp in change_points:
        segment = param_series[prev:cp]
        if len(segment) < 2:
            continue
        # 计算该段数据的均方误差
        mse = np.mean((segment - np.mean(segment))**2)
        errors.append(mse)
        prev = cp
    return np.mean(errors) if errors else 0


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
        
            # 新增误差计算
        if intervals is None or len(intervals) == 0:
            return None, None

        # 计算每个分段的误差
        mse_error = calculate_segment_errors(param_series, existing_indices)
        return intervals, existing_indices, mse_error
        
    except Exception as e:
        print(f"变点检测异常：{str(e)}")
        return None

def get_recommended_interval(intervals, errors, fallback=3600):
    """带误差评估的推荐"""
    if intervals is None:
        return fallback, None
    
    try:
        mode_val = mode(intervals).mode
        # 找到对应众数间隔的误差（可能有多个相同众数，取平均）
        mode_indices = np.where(intervals == mode_val)
        avg_error = np.mean([errors[i] for i in mode_indices])
        return mode_val, avg_error
    except:
        median_val = np.median(intervals).astype(int)
        return median_val, np.median(errors)

def simple_interval_detection(folder_path, initial_interval='10T'):
    # 数据加载（复用原有函数）
    dfs = load_and_preprocess_data(folder_path)
    time_grid = calculate_common_timeline(dfs, initial_interval)
    polar_dfs = convert_to_polar_coordinates(dfs, time_grid)
    param_df = calculate_parameters(polar_dfs, time_grid, initial_interval)
    
    # 分别检测两个参数
    speed_intervals,valid_cp_speed, speed_errors  = detect_single_param_interval(
        param_df['speed_mps'].values, 
        param_df['start_time'].values,
        penalty_value_velocity
    )
    angular_intervals, valid_cp_angular, angular_errors  = detect_single_param_interval(
        param_df['angular_speed_degps'].values,
        param_df['start_time'].values,
        penalty_value_angular
    )
    
    # 带误差的推荐
    speed_rec, speed_err = get_recommended_interval(speed_intervals, speed_errors, 7200)
    angular_rec, angular_err = get_recommended_interval(angular_intervals, angular_errors, 3600)
    
    # 综合误差（加权平均）
    valid_recs = []
    if speed_rec: valid_recs.append( (speed_rec, speed_err) )
    if angular_rec: valid_recs.append( (angular_rec, angular_err) )
    
    if valid_recs:
        final_rec = min([rec for rec in valid_recs])
        
        # 找到最终推荐对应的误差
        final_error = next((rec[1] for rec in valid_recs if rec == final_rec), None)
    else:
        final_rec, final_error = None, None
    
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
    

    # 转换为可读格式（新增误差字段）
    def format_output(seconds, error):        
        if seconds is None: return None
        readable = sec_to_readable(seconds)
        return f"{readable} (±{error:.2e})" if error else readable
    
    plot_dual_detection(param_df,valid_cp_speed,valid_cp_angular)
    final_rec = final_rec[0] if final_rec else None
    return {
        'speed': format_output(speed_rec, speed_err),
        'angular': format_output(angular_rec, angular_err),
        'final': format_output(final_rec, final_error)
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
    速度推荐间隔: {result['speed']}
    角速度推荐间隔: {result['angular']}
    最终推荐值: {result['final']}
    """)
