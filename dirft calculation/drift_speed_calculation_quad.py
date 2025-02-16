# 读取指定文件夹下的四个海洋浮标的漂移轨迹数据（csv）格式。并且做以下计算：
# 以用户指定的时间间隔（如1 h），计算四个海洋浮标所组成的四边形的漂移速度大小、漂移速度方向、旋转角速度大小（以四个海洋浮标的组成的四边形的重心为旋转中心）。输出每段时间内的结果。
from drift_utils import *

def main():
    folder_path = r"D:\Working_Project\Arctic_2024_Shuhang\Data\冰上GNSS控制点数据\0830_冰基长基线浮标GNSS_O文件转换\解算数据"
    interval = '10min'   
    
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