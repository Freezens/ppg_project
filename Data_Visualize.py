"""

test data set, train data set 의 시각화 구현 코드
peak_find()는 들어있지 않음

"""


# pip install plotly
import plotly as plt
import plotly.graph_objects as go
import numpy as np
from scipy.signal import resample

# .txt 데이터 파일 불러오기
def load_data_txt(file_path, HZ):
    with open(file_path, "r") as file:
        data = [float(line.strip()) for line in file]
    x = [i / HZ for i in range(len(data))]

    return data, x

def load_data_npy(file_path, data_size):
    data = np.load(file_path)

    return data

# Test Set 시각화 클래스
class TestSetVisualizer:
    def __init__ (self, dataset_folder = "PPG Test Dataset"):
        self.dataset_folder = dataset_folder
        self.files = (
            "Empty",
            "01_atrialfibrillation_2760_3660_3800_6930_7200_7490_7650.txt",
            "02_bradycardia_2760_4397_4762_5600_5787_7555_7805.txt",
            "03_bradycardia_2240_3810_4110_6810_7120_8890_9120.txt",
            "04_tachycardia_2170_3870_4370_5310_5940_8330_9005.txt",
            "05_atrialfibrillation_3040_3810_3957_4920_5040_5750_5890.txt",
            "06_bradycardia_2360_3760_4050_4780_5050_5450_5730.txt",
            "07_atrialfibrillation_2425_2730_2920_3760_3950_5000_5180.txt",
            "08_techycardia_1710_2450_2850_3570_3960_4800_5290.txt"
        )

        self.anomaly = [
            [0, 2760, 2760, 2240, 2170, 3040, 2360, 2425, 1710],
            [0, 3660, 4397, 3810, 3870, 3810, 3760, 2730, 2450],
            [0, 3800, 4762, 4110, 4370, 3957, 4050, 2920, 2850],
            [0, 6930, 5600, 6810, 5310, 4920, 4780, 3760, 3570],
            [0, 7200, 5787, 7120, 5940, 5040, 5050, 3950, 3960],
            [0, 7490, 7555, 8890, 8330, 5750, 5450, 5000, 4800],
            [0, 7650, 7805, 9120, 9005, 5890, 5730, 5180, 5290]
        ]

        self.select_file = 0
        self.data = []
        self.x = []
        self.HZ = 100

    def select_file_index(self):
        while self.select_file < 1 or self.select_file > 8:
            try:
                self.select_file = int(input("Select case num (1~8) : "))
            except ValueError:
                print("숫자를 입력하세요.")
    
    def load_data(self):
        file_path = f"{self.dataset_folder}/{self.files[self.select_file]}"
        self.data, self.x = load_data_txt(file_path, self.HZ)

    def plot(self):
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=self.x,
            y=self.data,
            mode='lines+markers',
            marker=dict(size=4),
            name="Signal"
        ))

        for c in [1, 3, 5]:
            start = self.anomaly[c][self.select_file]
            end = self.anomaly[c+1][self.select_file]

            highlight_x = self.x[start:end+1]
            highlight_y = self.data[start:end+1]

            fig.add_trace(go.Scatter(
                x=highlight_x,
                y=highlight_y,
                mode='lines+markers',
                marker=dict(size=4, color='red'),
                name=f"Anomaly ({start}-{end})"
            ))

        fig.update_layout(
            title=f"Data Visualization type.{self.select_file}",
            xaxis_title="Time (seconds)",
            yaxis_title="Value",
            xaxis=dict(
                rangeslider=dict(visible=True),
                type="linear"
            ),
            width=3000,
            height=500
        )

        fig.show()

    def run(self):
        self.select_file_index()
        self.load_data()
        self.plot()

class TrainSetVisualizer:
    def __init__(self):
        self.file_path = "ppg_classification.npy"
        self.sample_rate = 100
        self.segment_size = 6000
        self.x = np.arange(self.segment_size) / self.sample_rate

        self.arr1 = []  # 정상 환자 data
        self.arr2 = []  # 부정맥 환자 data

        self.normal_segments_resampled = []
        self.abnormal_segments_resampled = []
    
    def peaks_find(self, arr, Block_size = 10, prominence = 3):
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
    
    def get_peaks(self, y_values):
        peaks = self.peaks_find(y_values)
        peak_x = [self.x[p[0]] for p in peaks]
        peak_y = [p[1] for p in peaks]
        return peak_x, peak_y

    def load_data(self):
        data = np.load(self.file_path)
        self.arr1 = data[:10, :]   # shape (10, 30000)
        self.arr2 = data[10:, :]   # shape (10, 30000)

    def slicing_data(self):
        num_segments = 5  # 30000 / 6000
        segments_1 = []
        segments_2 = []

        for i in range(num_segments):
            start = i * self.segment_size
            end = start + self.segment_size
            segments_1.append(self.arr1[:, start:end])
            segments_2.append(self.arr2[:, start:end])

        self.arr1 = segments_1  # → list of 2D arrays
        self.arr2 = segments_2

    def get_segment(self, group: str, segment_idx: int, row_idx: int):
        """
        group: 'normal' or 'abnormal'
        segment_idx: 0~4
        row_idx: 환자 인덱스 (0~9 for normal, 0~39 for abnormal)
        """
        if group == 'normal':
            return self.x, self.arr1[segment_idx][row_idx]
        else:
            return self.x, self.arr2[segment_idx][row_idx]
        
    def resampling_data(self):

        for group_name, data_group in [('normal', self.arr1), ('abnormal', self.arr2)]:
            resampled_group = []

            for segment in data_group:  # segment: shape (10, 6000)
                for patient_idx in range(segment.shape[0]):
                    y = segment[patient_idx]
                    peaks = self.peaks_find(y)

                    # 최소 5개 이상 피크가 있어야 처리 가능
                    if len(peaks) < 5:
                        continue

                    # 피크 인덱스만 추출
                    peak_indices = [p[0] for p in peaks]

                    # 5개씩 슬라이딩 윈도우로 구간을 선택
                    for i in range(len(peak_indices) - 4):
                        start_idx = peak_indices[i]
                        end_idx = peak_indices[i + 4]

                        if end_idx - start_idx < 10:  # 너무 짧은 경우 무시
                            continue

                        segment_slice = y[start_idx:end_idx + 1]
                        # 100개로 리샘플링
                        resampled = resample(segment_slice, 100)
                        resampled_group.append(resampled)

            if group_name == 'normal':
                self.normal_segments_resampled = resampled_group
            else:
                self.abnormal_segments_resampled = resampled_group

    def plot(self):
        fig = go.Figure()

        # 기본 표시: 정상 환자 0번, 세그먼트 0
        x, y = self.get_segment(group='normal', segment_idx=0, row_idx=0)
        peak_x, peak_y = self.get_peaks(y)


        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines+markers',
            marker=dict(size=4),
            name="Normal Segment 0 - Patient 0"
        ))

         # 피크
        fig.add_trace(go.Scatter(
            x=peak_x,
            y=peak_y,
            mode='markers',
            marker=dict(size=8, color='red'),
            name='Peaks'
        ))

        # 드롭다운 버튼 생성
        dropdown_buttons = []

        # 정상 환자
        for s in range(len(self.arr1)):
            for r in range(self.arr1[0].shape[0]):
                x_seg, y_seg = self.get_segment('normal', s, r)
                peak_x, peak_y = self.get_peaks(y_seg)
                dropdown_buttons.append({
                    "args": [
                        {"x": [x_seg, peak_x],
                         "y": [y_seg, peak_y]}
                    ],
                    "label": f"Normal: Patient {1 + r + s * 10}",
                    "method": "update"
                })

        # 부정맥 환자
        for s in range(len(self.arr2)):
            for r in range(self.arr2[0].shape[0]):
                x_seg, y_seg = self.get_segment('abnormal', s, r)
                peak_x, peak_y = self.get_peaks(y_seg)
                dropdown_buttons.append({
                    "args": [
                        {"x": [x_seg, peak_x],
                         "y": [y_seg, peak_y]}
                    ],
                    "label": f"Abnormal: Patient {1 + r + s * 10}",
                    "method": "update"
                })

        fig.update_layout(
            title="100Hz Sampled PPG Data with Peak Detection",
            xaxis_title="Time (seconds)",
            yaxis_title="PPG Value",
            xaxis=dict(
                rangeslider=dict(visible=True),
                type="linear"
            ),
            width=3000,
            height=500,
            updatemenus=[{
                "buttons": dropdown_buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.1,
                "xanchor": "left",
                "y": 1.15,
                "yanchor": "top"
            }]
        )
        return fig

    def plot_resampled(self):
        fig = go.Figure()

        # 기본 표시: 정상 리샘플링 데이터 첫 번째
        x = np.arange(100)
        y = self.normal_segments_resampled[0] if self.normal_segments_resampled else np.zeros(100)

        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines+markers',
            marker=dict(size=4),
            name="Resampled Normal Segment 0"
        ))

        dropdown_buttons = []

        # 정상 세그먼트
        for i, seg in enumerate(self.normal_segments_resampled):
            dropdown_buttons.append({
                "args": [{"y": [seg], "x": [x]}],
                "label": f"Normal {i}",
                "method": "update"
            })

        # 부정맥 세그먼트
        for i, seg in enumerate(self.abnormal_segments_resampled):
            dropdown_buttons.append({
                "args": [{"y": [seg], "x": [x]}],
                "label": f"Abnormal {i}",
                "method": "update"
            })

        fig.update_layout(
            title="Resampled PPG Segments (Heartbeat-normalized)",
            xaxis_title="Normalized Time (0 ~ 100 samples)",
            yaxis_title="PPG Value",
            xaxis=dict(
                range=[0, 100],
                type="linear"
            ),
            width=1000,
            height=400,
            updatemenus=[{
                "buttons": dropdown_buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.1,
                "xanchor": "left",
                "y": 1.2,
                "yanchor": "top"
            }]
        )

        return fig

    def save_data(self, normal_path='resampled_normal.npy', abnormal_path='resampled_abnormal.npy'):

        normal_array = np.array(self.normal_segments_resampled)
        abnormal_array = np.array(self.abnormal_segments_resampled)

        np.save(normal_path, normal_array)
        np.save(abnormal_path, abnormal_array)


    def run(self):
        self.load_data()
        self.slicing_data()
        self.resampling_data()
        self.save_data()
        #print(len(self.normal_segments_resampled))
        #print(len(self.abnormal_segments_resampled))
        #fig = self.plot_resampled()
        #fig.show()



if __name__ == "__main__":
    visualizer = TrainSetVisualizer()
    visualizer.run()