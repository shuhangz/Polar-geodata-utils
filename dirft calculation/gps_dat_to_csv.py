import csv
import os
from datetime import datetime, timedelta
import math

def mjd_sod_to_datetime(mjd, sod):
    days_since_mjd0 = mjd + 2400000.5 - 2440587.5
    base_date = datetime(1970, 1, 1)
    total_seconds = days_since_mjd0 * 86400 + sod
    return base_date + timedelta(seconds=total_seconds)

def haversine(lon1, lat1, lon2, lat2):
    R = 6371000  # 地球半径，单位为米
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def calculate_bearing(lon1, lat1, lon2, lat2):
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_lambda = math.radians(lon2 - lon1)
    y = math.sin(delta_lambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)
    bearing = math.atan2(y, x)
    return (math.degrees(bearing) + 360) % 360

def read_dat_file(file_path, interval):
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        in_header = True
        for line in lines:
            if line.strip() == 'END OF HEADER':
                in_header = False
                continue
            if not in_header and line.startswith('*'):
                continue
            if not in_header and line.strip():
                parts = line.split()
                if parts[2] == '*':
                    parts.pop(2)  # 移除第三列
                mjd = int(parts[0])
                sod = float(parts[1])
                latitude = float(parts[5])
                longitude = float(parts[6])
                if longitude > 180:
                    longitude = longitude - 360
                height = float(parts[7])
                pdop = float(parts[-1])
                timestamp = mjd_sod_to_datetime(mjd, sod)
                data.append((timestamp, longitude, latitude, height, pdop))

    filtered_data = []
    last_timestamp = None
    for timestamp, longitude, latitude, height, pdop in data:
        if last_timestamp is None or (timestamp - last_timestamp).total_seconds() >= interval:
            filtered_data.append((timestamp, longitude, latitude, height, pdop))
            last_timestamp = timestamp

    return filtered_data

def calculate_speed_and_direction(data):
    results = []
    for i in range(1, len(data)):
        t1, lon1, lat1, height1, pdop1 = data[i - 1]
        t2, lon2, lat2, height2, pdop2 = data[i]
        distance = haversine(lon1, lat1, lon2, lat2)
        time_diff = (t2 - t1).total_seconds()
        speed = distance / time_diff if time_diff > 0 else 0
        direction = calculate_bearing(lon1, lat1, lon2, lat2)
        results.append((t2, lon2, lat2, height2, pdop2, speed, direction))
    return results

def apply_moving_average(data, window_size):
    smoothed_data = []
    for i in range(len(data)):
        if i < window_size - 1:
            smoothed_data.append(data[i])
        else:
            avg_speed = sum(d[5] for d in data[i - window_size + 1:i + 1]) / window_size
            avg_direction = sum(d[6] for d in data[i - window_size + 1:i + 1]) / window_size
            smoothed_data.append((data[i][0], data[i][1], data[i][2], data[i][3], data[i][4], avg_speed, avg_direction))
    return smoothed_data

def write_csv_file(data, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['datetime', 'longitude', 'latitude', 'height', 'PDOP', 'speed', 'direction'])
        for row in data:
            writer.writerow(row)

if __name__ == "__main__":
    input_folder = r"D:\Work\Project data\Arctic_2024\Data\冰上GNSS控制点数据\20240904长期冰站GNSS观测数据\20240904长期冰站定位结果\250"  # 替换为实际的文件夹路径
    interval = 1  # 输出频率，单位为秒，可以根据需要修改
    window_size = 5  # 滑动平均窗口大小

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.dat'):
            input_file = os.path.join(input_folder, file_name)
            output_file = os.path.splitext(input_file)[0] + '.csv'
            data = read_dat_file(input_file, interval)
            speed_and_direction_data = calculate_speed_and_direction(data)
            smoothed_data = apply_moving_average(speed_and_direction_data, window_size)
            write_csv_file(smoothed_data, output_file)
            print(f"数据已成功写入 {output_file}")