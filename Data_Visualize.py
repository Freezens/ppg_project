"""

test data set, train data set 의 시각화 구현 코드
peak_find()는 들어있지 않음

"""


# pip install plotly
import plotly.graph_objects as go
import numpy as np

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

        self.arr = []

    def load_data():
        


if __name__ == "__main__":
    visualizer = TestSetVisualizer()
    visualizer.run()
