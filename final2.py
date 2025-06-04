import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import tensorflow as tf
from sklearn.metrics import roc_curve, auc, f1_score, recall_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, Dense, Dropout, Conv1D, GlobalMaxPooling1D, BatchNormalization
import random

random.seed(42)

class SelfAttention1D(Layer):
    def __init__(self, **kwargs):
        super(SelfAttention1D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.d_k = input_shape[-1]
        self.query = self.add_weight(shape=(self.d_k, self.d_k), initializer='glorot_uniform', trainable=True)
        self.key = self.add_weight(shape=(self.d_k, self.d_k), initializer='glorot_uniform', trainable=True)
        self.value = self.add_weight(shape=(self.d_k, self.d_k), initializer='glorot_uniform', trainable=True)

    def call(self, x):
        Q = tf.matmul(x, self.query)
        K = tf.matmul(x, self.key)
        V = tf.matmul(x, self.value)

        score = tf.matmul(Q, K, transpose_b=True)
        score = score / tf.math.sqrt(tf.cast(self.d_k, tf.float32))
        weights = tf.nn.softmax(score, axis=-1)
        output = tf.matmul(weights, V)
        return output
    

# 0. 모델 불러오기
model_cnn = load_model("cnn_lstm_model.h5")
model_fft = load_model("fft_model.h5")

weight_cnn = 1.7
weight_fft = 0.3 - weight_cnn

# 1. 테스트 데이터 불러오기
test_n = np.load('resampled_normal.npy')      # 정상 샘플
test_abn = np.load('resampled_abnormal.npy')  # 이상 샘플

indices = list(range(len(test_abn)))
test_abn_indices = random.sample(indices, 500)

test_abn_500 = test_abn[test_abn_indices]
train_abn = np.delete(test_abn, test_abn_indices, axis=0)

# 2. X, y 구성
X_test = np.concatenate([test_n, train_abn], axis=0)
y_test = np.array([0]*len(test_n) + [1]*len(train_abn))  # 0: 정상, 1: 이상

print (X_test.shape)
# 3. 전처리: 정규화 및 reshape
scaler = MinMaxScaler()
X_test_scaled = np.array([scaler.fit_transform(x.reshape(-1, 1)).flatten() for x in X_test])
X_test_scaled = X_test_scaled.reshape(-1, 100, 1)  # (샘플 수, 길이, 채널)

# 4. 개별 모델 예측 확률
y_pred_prob_cnn = model_cnn.predict(X_test_scaled)
y_pred_prob_fft = model_fft.predict(X_test_scaled)

# 5. 2채널 입력으로 메타 모델용 입력 생성
combined_input = np.hstack([weight_cnn * y_pred_prob_cnn, weight_fft * y_pred_prob_fft])  # (샘플 수, 2)
X_meta_all = combined_input.reshape(-1, 1, 2)  # (샘플 수, 길이=1, 채널=2)

# 6. 학습용/검증용 분할
X_meta_train, X_meta_val, y_meta_train, y_meta_val = train_test_split(
    X_meta_all, y_test, test_size=0.2, random_state=42
)

print (X_meta_all.shape)
# 7. 메타 모델 정의
meta_model = Sequential([
    Conv1D(64, kernel_size=1, activation='relu', input_shape=(1, 2)),
    BatchNormalization(),
    #SelfAttention1D(), 
    GlobalMaxPooling1D(),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# 8. 훈련
meta_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
meta_model.fit(X_meta_train, y_meta_train, epochs=2, batch_size=8, verbose=1)

# 9. 평가
loss, acc = meta_model.evaluate(X_meta_val, y_meta_val)
print(f"Meta-Model Accuracy: {acc:.4f}")

# 10. 예측 확률
y_meta_prob = meta_model.predict(X_meta_all).flatten()

# 11. ROC Curve 계산
fpr_cnn, tpr_cnn, _ = roc_curve(y_test, y_pred_prob_cnn)
auc_cnn = auc(fpr_cnn, tpr_cnn)

fpr_fft, tpr_fft, _ = roc_curve(y_test, y_pred_prob_fft)
auc_fft = auc(fpr_fft, tpr_fft)

fpr_meta, tpr_meta, _ = roc_curve(y_test, y_meta_prob)
auc_meta = auc(fpr_meta, tpr_meta)

# 12. ROC 시각화
plt.figure(figsize=(8, 6))
plt.plot(fpr_cnn, tpr_cnn, label=f"CNN-LSTM (AUC = {auc_cnn:.2f})", linestyle='--', color='blue')
plt.plot(fpr_fft, tpr_fft, label=f"FFT Model (AUC = {auc_fft:.2f})", linestyle='-.', color='green')
plt.plot(fpr_meta, tpr_meta, label=f"Meta Model (AUC = {auc_meta:.2f})", linewidth=2.2, color='red')

plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

# 13. 확률 히스토그램 시각화
n0, bins0, _ = plt.hist(y_meta_prob[y_test == 0], bins=100, alpha=0.5, label='Normal (0)', color='green')
n1, bins1, _ = plt.hist(y_meta_prob[y_test == 1], bins=100, alpha=0.5, label='Abnormal (1)', color='red')

for i in range(len(n0)):
    if n0[i] > 0:
        plt.text((bins0[i] + bins0[i+1]) / 2, n0[i], str(int(n0[i])), ha='center', va='bottom', fontsize=8, color='green')

for i in range(len(n1)):
    if n1[i] > 0:
        plt.text((bins1[i] + bins1[i+1]) / 2, n1[i], str(int(n1[i])), ha='center', va='bottom', fontsize=8, color='red')

plt.title("Histogram of Predicted Probabilities by Class")
plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.xlim(0, 1)
plt.xticks(np.linspace(0, 1, 21))
plt.tight_layout()
plt.show()

#recall, f1
max_prob = np.max(y_meta_prob)
threshold = max_prob * 0.9

y_meta_pred = (y_meta_prob >= threshold).astype(int)

f1 = f1_score(y_test, y_meta_pred)
recall = recall_score(y_test, y_meta_pred)
print(f"Meta-Model F1 Score: {f1:.4f}")
print(f"Meta-Model Recall: {recall:.4f}")







test_n = np.load('normal_test_data.npy')      # 정상 테스트 샘플
test_abn = np.load('abnormal_test_data.npy')  # 비정상 테스트 샘플

test_abn = np.concatenate([test_abn, test_abn_500], axis=0)
X_test = np.concatenate([test_n, test_abn], axis=0)
y_test = np.array([0]*len(test_n) + [1]*len(test_abn))

# --- 전처리: 정규화 및 reshape ---
scaler = MinMaxScaler()
X_test_scaled = np.array([scaler.fit_transform(x.reshape(-1, 1)).flatten() for x in X_test])
X_test_scaled = X_test_scaled.reshape(-1, 100, 1)  # shape 맞추기

# --- 개별 모델 예측 ---
y_pred_prob_cnn = model_cnn.predict(X_test_scaled)
y_pred_prob_fft = model_fft.predict(X_test_scaled)

# --- 메타 모델 입력 생성 ---
combined_input = np.hstack([weight_cnn * y_pred_prob_cnn, weight_fft * y_pred_prob_fft])  # (샘플 수, 2)
X_meta_all = combined_input.reshape(-1, 1, 2)

# --- 메타 모델 예측 ---
y_meta_prob = meta_model.predict(X_meta_all).flatten()

# --- F1, Recall 계산 ---
max_prob = np.max(y_meta_prob)
threshold = max_prob * 0.9
y_meta_pred = (y_meta_prob >= threshold).astype(int)

f1 = f1_score(y_test, y_meta_pred)
recall = recall_score(y_test, y_meta_pred)
print(f"Meta-Model F1 Score: {f1:.4f}")
print(f"Meta-Model Recall: {recall:.4f}")

# --- 확률 히스토그램 시각화 ---
n0, bins0, _ = plt.hist(y_meta_prob[y_test == 0], bins=100, alpha=0.5, label='Normal (0)', color='green')
n1, bins1, _ = plt.hist(y_meta_prob[y_test == 1], bins=100, alpha=0.5, label='Abnormal (1)', color='red')

for i in range(len(n0)):
    if n0[i] > 0:
        plt.text((bins0[i] + bins0[i+1]) / 2, n0[i], str(int(n0[i])), ha='center', va='bottom', fontsize=8, color='green')

for i in range(len(n1)):
    if n1[i] > 0:
        plt.text((bins1[i] + bins1[i+1]) / 2, n1[i], str(int(n1[i])), ha='center', va='bottom', fontsize=8, color='red')

plt.title("Histogram of Predicted Probabilities by Class")
plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.xlim(0, 1)
plt.xticks(np.linspace(0, 1, 21))
plt.tight_layout()
plt.show()

from tensorflow.keras.utils import plot_model

plot_model(meta_model, to_file='meta_model.svg', show_shapes=True, show_layer_names=True)

