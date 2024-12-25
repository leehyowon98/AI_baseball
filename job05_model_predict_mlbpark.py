import pickle # 타입변환 없이 binary 코드로 저장

import pandas as pd #dataFrame 문자열을 맞는 데이터타입으로 변환
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
#한국어 토큰나이저
# jpype1 설치
from konlpy.tag import Okt,Kkma  #open korean tokenizer,java 기반
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.api.models import load_model

# 데이터 로드 및 전처리
df = pd.read_csv('./crawling_data/mlbpark_titles_303page.csv')
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)

print(df.head())
df.info()

print(df['category'].value_counts())

# 공백 제거
df['category'] = df['category'].str.strip()
X = df['titles']
Y = df['category']

# 레이블 인코더 재생성
encoder = LabelEncoder()
encoder.fit(Y)  # 현재 데이터셋에 맞게 인코더 적합

# 레이블 변환
label = encoder.classes_
print("Labels:", label)

# 레이블을 숫자로 인코딩
labeled_y = encoder.transform(Y)

# 원-핫 인코딩 (10개 클래스에 맞게)
onehot_Y = to_categorical(labeled_y, num_classes=10)
print("One-hot encoded labels shape:", onehot_Y.shape)

# x 문장데이터 처리, 불용어 제거
okt = Okt()
X_morphs = []
for title in X:
    X_morphs.append(okt.morphs(title, stem=True))

# stopwords 불용어 다운로드
stopwords = pd.read_csv('./crawling_data/stopwords.csv', index_col=0)

# 불용어 제거
X_cleaned = []
for sentence in X_morphs:
    words = [word for word in sentence if len(word) > 1 and word not in list(stopwords['stopword'])]
    X_cleaned.append(' '.join(words))

# 토큰나이징
with open('models/MP_KBO_token_max_37.pickle', 'rb') as f:
    token = pickle.load(f)
tokened_X = token.texts_to_sequences(X_cleaned)

# 토큰 길이 제한
for i in range(len(tokened_X)):
    if len(tokened_X[i]) > 37:
        tokened_X[i] = tokened_X[i][:37]

# 패딩
X_pad = pad_sequences(tokened_X, maxlen=37)

# 모델 로드
model = load_model('./models/MP_KBO_category_classification_model_0.6137456893920898.h5')

# 예측
preds = model.predict(X_pad)

# 모든 클래스를 포함하도록 레이블 설정
all_labels = ['DOOSAN', 'HANWHA', 'KIA', 'KIWOOM', 'KT', 'LG', 'LOTTE', 'NC', 'SAMSUNG', 'SSG']

# 예측 코드 수정
predicts = []
for pred in preds:
    most_index = np.argmax(pred)
    most = all_labels[most_index]
    pred[most_index] = 0
    second_index = np.argmax(pred)
    second = all_labels[second_index]
    predicts.append([most, second])

df['predict'] = predicts

print(df.head(30))

# 모델 평가
score = model.evaluate(X_pad, onehot_Y)
print("Model Evaluation Score:", score[1])

# 예측 정확도 계산
df['OX'] = 0
for i in range(len(df)):
    if df.loc[i, 'category'] in df.loc[i, 'predict']:
        df.loc[i, 'OX'] = 1

print("Prediction Accuracy:", df['OX'].mean())