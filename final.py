#필터 16개
import ppgcnn
import ppgfft
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 0. 모델 불러오기
model_cnn = ppgcnn.model
model_fft = ppgfft.model

# 1. 테스트 데이터 불러오기
test_n = np.load('normal_test_data.npy')      # 정상 샘플
test_abn = np.load('abnormal_test_data.npy')  # 이상 샘플

# 2. X, y 구성
X_test = np.concatenate([test_n, test_abn], axis=0)
y_test = np.array([0]*len(test_n) + [1]*len(test_abn))  # 0: 정상, 1: 이상

# 3. 전처리: 정규화 및 reshape
scaler = MinMaxScaler()
X_test_scaled = np.array([scaler.fit_transform(x.reshape(-1, 1)).flatten() for x in X_test])
X_test_scaled = X_test_scaled.reshape(-1, 100, 1)  # (샘플 수, 길이, 채널)

y_pred_prob_cnn = model_cnn.predict(X_test_scaled)
y_pred_prob_fft = model_fft.predict(X_test_scaled)

y_pred_cnn = (y_pred_prob_cnn >= 0.96).astype(int).flatten()
y_pred_fft = (y_pred_prob_fft >= 0.5).astype(int).flatten()

print (y_pred_prob_cnn)
print (y_pred_prob_fft)