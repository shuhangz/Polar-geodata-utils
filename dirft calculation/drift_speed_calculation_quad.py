# 读取指定文件夹下的四个海洋浮标的漂移轨迹数据（csv）格式。并且做以下计算：
# 以用户指定的时间间隔（如1 h），计算四个海洋浮标所组成的四边形的漂移速度大小、漂移速度方向、旋转角速度大小（以四个海洋浮标的组成的四边形的重心为旋转中心）。输出每段时间内的结果。
from drift_utils import *
import matplotlib.pyplot as plt

def main():
    # folder_path = r"D:\Working_Project\Arctic_2024_Shuhang\Data\冰上GNSS控制点数据\0830_冰基长基线浮标GNSS_O文件转换\解算数据"
    folder_path = r"D:\Working_Project\Arctic_2024_Shuhang\Data\冰上GNSS控制点数据\20240904长期冰站GNSS观测数据\20240904长期冰站定位结果\249"
    interval = '600s'   
    
    dfs = load_and_preprocess_data(folder_path)
    time_grid = calculate_common_timeline(dfs, interval)
    polar_dfs = convert_to_polar_coordinates(dfs, time_grid)
    results = calculate_parameters(polar_dfs, time_grid, interval)
    
    print("\n计算结果：")
    print(results.to_string(index=False))
    results.to_csv('polar_results.csv', index=False)
    print("\n结果已保存到 polar_results.csv")


    # 绘图
    plt.figure(figsize=(12, 9))
    # 绘制速度图
    plt.subplot(3, 1, 1)
    plt.plot(results['start_time'], results['speed_mps'], label='Speed (m/s)')
    plt.xlabel('Start Time')
    plt.ylabel('Speed (m/s)')
    plt.title('Drift Speed Over Time')
    plt.legend()

    # 绘制角速度图
    plt.subplot(3, 1, 2)
    plt.plot(results['start_time'], results['angular_speed_degps'], label='Angular Speed (deg/s)', color='orange')
    plt.xlabel('Start Time')
    plt.ylabel('Angular Speed (deg/s)')
    plt.title('Angular Speed Over Time')
    plt.legend()

    # 绘制方向图
    plt.subplot(3, 1, 3)
    plt.plot(results['start_time'], results['direction_deg'], label='Direction (deg)', color='green')
    plt.xlabel('Start Time')
    plt.ylabel('Direction (deg)')
    plt.title('Drift Direction Over Time')
    plt.legend()

    # 绘制漂流轨迹图
    plt.figure(figsize=(9, 9))
    for df in polar_dfs:
        df = df.iloc[1:]
        plt.scatter(df['x'], df['y'], c=results['speed_mps'], cmap='viridis', label='Drift Trajectory')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Drift Trajectory')
    plt.colorbar(label='Speed (m/s)')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()