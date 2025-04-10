"""

train data set 시각화 및 peak 찾기 구현
train data set 은 총 1000개, 각각 300개의 데이터를 가짐

"""

# pip install plotly
import numpy as np
import plotly.graph_objects as go

# .txt 형식 데이터 불러오기
def load_data_txt(file_path, HZ):
    with open(file_path, "r") as file:
        data = [float(line.strip()) for line in file]
    x = [i / HZ for i in range(len(data))]

    return data, x


# .npy 형식 데이터 불러오기
def load_data_npy(file_path):
    arr = np.load(file_path)
    flatten_arr = arr.flatten()
    split_arr = np.split(flatten_arr, 1000)

    return split_arr

"""

peak_find() 함수.
이웃한 값 간의 차이가 굉장히 작아 기본적인 peak_find 방식으로는 peak를 찾기 어려움.
이를 해결할 함수 구현. 아래는 기본적인 peak_find 함수.
반환형은 (x, y) 쌍의 배열 

"""

def peaks_find(arr, prominence = 0):
    peaks = []

    for i in range(1, len(arr) - 1):
        if arr[i-1] < arr[i] > arr[i+1]:
            left_base = max(arr[:i]) if i > 0 else arr[i]
            right_base = max(arr[i+1:]) if i < len(arr)-1 else arr[i]
            reference_level = max(left_base, right_base)

            peak_prominence = arr[i] - reference_level

            if peak_prominence >= prominence:
                peaks.append((i, arr[i]))

    return peaks

"""

개선된 peak_find 함수.
일정 간격으로 블록을 생성, 해당 블록 내에서의 최댓값을 구한 후,
각 블록의 평균값을 앞 뒤 블록과 비교하여 해당 값이 peak인지 아닌지 판단
양 블록과 일정수치 이상 차이나지 않으면 peak가 아닌것으로 간주

"""

def peaks_find_2(arr, Block_size = 10, prominence = 0.5):
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


# Train data set 시각화 클래스. 해당 클래스의 run() 함수 사용.
class TrainDataSet_Visualize:
    def __init__(self, file_path = 'ppg_train.npy'):
        self.file_path = file_path  # 파일 이름 수정 필요시 해당 부분 수정
        self.split_size = 300       # 샘플 별로 나누는 크기 수정 필요 시 해당 부분 수정
        self.split_array = []       # split 된 데이터 저장
        self.flatten_data = []      # flatten 된 데이터 저장
        self.peaks = []             # peaks_find() 함수를 이용해 뽑아낸 peaks들 저장

    # 데이터 불러오기 
    def load_data(self):
        self.split_array = load_data_npy(self.file_path)
        self.flatten_data = [chunk.flatten() for chunk in self.split_array]

    # peaks 찾기 및 총 peaks 수 출력 (나중에 완성되면 지울 것)
    def find_peaks(self):
        self.peaks = [peaks_find_2(arr, 5) for arr in self.flatten_data]
        print(sum(len(peaks) for peaks in self.peaks))

    # 시각화 함수
    def plot(self):
        fig = go.Figure()

        for i,flat in enumerate(self.flatten_data):
            fig.add_trace(
                go.Scatter(
                    y=flat,
                    mode='lines',
                    name=f'Data {i+1}',
                    visible=(i == 0)
                )
            )

            peak_indices = [p[0] for p in self.peaks[i]]
            peak_values = [p[1] for p in self.peaks[i]]
            fig.add_trace(
                go.Scatter(
                    x=peak_indices,
                    y=peak_values,
                    mode='markers',
                    marker=dict(color='red', size=6, symbol='circle'),
                    name=f'Peaks {i+1}',
                    visible=(i == 0)
                )
            )

        dropdown_buttons = [
            dict(
                label=f'Data {i+1}',
                method='update',
                args=[{
                    'visible': [j // 2 == i for j in range(2 * len(self.flatten_data))],
                    'title': f'Data {i+1}'
                }]
            )
            for i in range(len(self.flatten_data))
        ]

        fig.update_layout(
            title='Flattened Data Viewer',
            updatemenus=[dict(
                active=0,
                buttons=dropdown_buttons,
                x=1.1,
                y=1.15
            )],
            xaxis_title='Index',
            yaxis_title='Value',
            height=400
        )

        fig.show()

    def run(self):
        self.load_data()
        self.find_peaks()
        self.plot()


if __name__ == "__main__":
    visualizer = TrainDataSet_Visualize()
    visualizer.run()
