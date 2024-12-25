import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPool1D, LSTM, Dropout, Flatten, Dense, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# 데이터 로드
X_train = np.load('./crawling_data/MP_KBO_X_train_wordsize_10828_max_37.npy', allow_pickle=True)
Y_train = np.load('./crawling_data/MP_KBO_Y_train_wordsize_10828_max_37.npy', allow_pickle=True)
X_test = np.load('./crawling_data/MP_KBO_X_test_wordsize_10828_max_37.npy', allow_pickle=True)
Y_test = np.load('./crawling_data/MP_KBO_Y_test_wordsize_10828_max_37.npy', allow_pickle=True)

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

# 모델 정의
model = Sequential()
model.add(Embedding(10828, 300, input_length=37))
model.add(Conv1D(64, kernel_size=5, padding='same', activation='relu'))  # 필터 수 증가
model.add(BatchNormalization())  # 배치 정규화 추가
model.add(MaxPool1D(pool_size=2))  # 풀링 크기 조정
model.add(LSTM(128, activation='tanh', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64, activation='tanh', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64, activation='tanh'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

# 모델 컴파일
optimizer = Adam(learning_rate=0.001)  # 학습률 조정
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# 조기 종료 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# 모델 훈련
fit_hist = model.fit(X_train, Y_train, batch_size=128,
                     epochs=30, validation_data=(X_test, Y_test), callbacks=[early_stopping])

# 모델 평가
score = model.evaluate(X_test, Y_test, verbose=0)
print('Final test set accuracy', score[1])

# 모델 저장
model.save('./models/MP_KBO_category_classification_model_{}.h5'.format(
    fit_hist.history['val_accuracy'][-1]
))

# 정확도 시각화
plt.plot(fit_hist.history['val_accuracy'], label='val_accuracy')
plt.plot(fit_hist.history['accuracy'], label='accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()