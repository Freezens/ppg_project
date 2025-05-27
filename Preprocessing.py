"""

train data set 시각화 및 peak 찾기 구현
train data set 은 총 1000개, 각각 300개의 데이터를 가짐

"""

# pip install plotly
import numpy as np
import plotly.graph_objects as go
import plotly as plt
from scipy.signal import resample

files = (
    "01_atrialfibrillation_2760_3660_3800_6930_7200_7490_7650.txt",
    "02_bradycardia_2760_4397_4762_5600_5787_7555_7805.txt",
    "03_bradycardia_2240_3810_4110_6810_7120_8890_9120.txt",
    "04_tachycardia_2170_3870_4370_5310_5940_8330_9005.txt",
    "05_atrialfibrillation_3040_3810_3957_4920_5040_5750_5890.txt",
    "06_bradycardia_2360_3760_4050_4780_5050_5450_5730.txt",
    "07_atrialfibrillation_2425_2730_2920_3760_3950_5000_5180.txt",
    "08_techycardia_1710_2450_2850_3570_3960_4800_5290.txt"
    )

anomaly = [
    [2760, 2760, 2240, 2170, 3040, 2360, 2425, 1710],
    [3660, 4397, 3810, 3870, 3810, 3760, 2730, 2450],
    [3800, 4762, 4110, 4370, 3957, 4050, 2920, 2850],
    [6930, 5600, 6810, 5310, 4920, 4780, 3760, 3570],
    [7200, 5787, 7120, 5940, 5040, 5050, 3950, 3960],
    [7490, 7555, 8890, 8330, 5750, 5450, 5000, 4800],
    [7650, 7805, 9120, 9005, 5890, 5730, 5180, 5290]
    ]

# .txt 형식 데이터 불러오기
def load_data_txt(file_path, HZ):
    with open(file_path, "r") as file:
        data = [float(line.strip()) for line in file]
    x = [i / HZ for i in range(len(data))]

    return data, x

def peaks_find(arr, Block_size = 10, prominence = 3):
    max_in_block = [(0, arr[0], arr[0])]
    peaks = []
    for i in range(1, len(arr), Block_size):
        block = arr[i:i + Block_size]
        max_val = max(block)
        avg_val = np.mean(block)
        local_idx = np.argmax(block)
        absolute_idx = i + local_idx  # 전체 배열 기준 인덱스
        max_in_block.append((absolute_idx, max_val, avg_val))
        
    max_in_block.append((len(arr)-1, arr[-1], arr[-1]))

    for i in range(1, len(max_in_block)-1):
        if max_in_block[i][2] >= max_in_block[i-1][2] and max_in_block[i][2] > max_in_block[i+1][2] and (2*max_in_block[i][2] - (max_in_block[i-1][2]+max_in_block[i+1][2]))/2 > prominence:
            if max_in_block[i][0]>=3:
                peaks.append((max_in_block[i][0], max_in_block[i][1]))

    return peaks

def resample_txt_PPG_data(data, file_num):
    peaks = peaks_find(data)
    if len(peaks) < 5:
        return []
    
    peak_indices = [p[0] for p in peaks]
    normal_segments = []
    abnormal_segments = []

    for i in range(len(peak_indices) - 4):
        start_idx = peak_indices[i]
        end_idx = peak_indices[i+4]

        if end_idx - start_idx < 100:
            continue

        segment_slice = data[start_idx:end_idx+1]
        resampled = resample(segment_slice, 100)

        anomaly_range = [row[file_num] for row in anomaly]
        anomaly_check = False

        for j in range(3) :
            if start_idx <= anomaly_range[2*j+1] and end_idx >= anomaly_range[2*j+2] :
                anomaly_check = True
            if start_idx >= anomaly_range[2*j+1] and end_idx <= anomaly_range[2*j+2] :
                anomaly_check = True
        
        if anomaly_check == False :
            normal_segments.append(resampled)
        else:
            abnormal_segments.append(resampled)

    return np.array(normal_segments), np.array(abnormal_segments)

def save_data(file_path, file):
    save_file = np.array(file)
    np.save(file_path, file)


if __name__ == "__main__":

    data1 = []
    data2 = []

    for i in range(8):
        file_path = "PPG Test Dataset/" + files[i]
        data, x = load_data_txt(file_path=file_path, HZ=100)
        normal_data, abnormal_data = resample_txt_PPG_data(data, i)
        data1.append(normal_data)
        data2.append(abnormal_data)

    normal_all = np.concatenate(data1, axis=0)
    abnormal_all = np.concatenate(data2, axis=0)

    save_data("normal_test_data.npy", normal_all)
    save_data("abnormal_test_data.npy", abnormal_all)

    loaded = np.load("normal_test_data.npy")
    print(loaded.shape)
    loaded = np.load("abnormal_test_data.npy")
    print(loaded.shape)
