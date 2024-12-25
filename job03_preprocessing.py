import pandas as pd  # pandas 라이브러리 임포트
import numpy as np  # numpy 라이브러리 임포트
from sklearn.model_selection import train_test_split  # 데이터 분할을 위한 함수 임포트
from sklearn.preprocessing import LabelEncoder  # 레이블 인코딩을 위한 클래스 임포트
from keras.utils import to_categorical  # 원-핫 인코딩을 위한 함수 임포트
import pickle  # 객체 직렬화를 위한 pickle 모듈 임포트
from konlpy.tag import Okt  # 한국어 형태소 분석을 위한 Okt 클래스 임포트
from tensorflow.keras.preprocessing.text import Tokenizer  # 텍스트 전처리를 위한 Tokenizer 클래스 임포트
from tensorflow.keras.preprocessing.sequence import pad_sequences  # 시퀀스 패딩을 위한 함수 임포트
##
# CSV 파일에서 데이터프레임 읽기
df = pd.read_csv('crawling_data/mlbpark_titles_with_category.csv')
df.drop_duplicates(inplace=True)      # 중복제거 (주석 처리됨)
df.reset_index(drop=True, inplace=True)  # 인덱스를 재설정하여 연속적인 인덱스 부여
print(df.head())  # 데이터프레임의 처음 5행 출력
df.info()  # 데이터프레임의 정보 출력
print(df.category.value_counts())  # 각 카테고리의 개수 출력

X = df['titles']  # 뉴스 제목을 X에 저장
Y = df['category']  # 카테고리를 Y에 저장

print(X[0])  # 첫 번째 뉴스 제목 출력
okt = Okt()  # Okt 형태소 분석기 객체 생성
okt_x = okt.morphs(X[0], stem=True)  # 첫 번째 뉴스 제목을 형태소 분석하여 어간 추출
print('okt : ' , okt_x)  # 형태소 분석 결과 출력
# kkma = Kkma()  # Kkma 형태소 분석기 객체 생성 (주석 처리됨)
# kkma_x = kkma.morphs(X[0])  # Kkma로 첫 번째 뉴스 제목을 형태소 분석
# print('kkma : ' , kkma_x)  # Kkma 형태소 분석 결과 출력
# exit()  # 코드 실행 중지 (주석 처리됨)

encoder = LabelEncoder()  # 레이블 인코더 객체 생성
labeled_y = encoder.fit_transform(Y)  # 카테고리를 정수로 변환
print(labeled_y[:3])  # 변환된 레이블의 처음 3개 출력

label = encoder.classes_  # 인코딩된 레이블의 클래스 이름 저장
print(label)  # 클래스 이름 출력

# 인코더 객체를 pickle로 저장
with open('./models/encoder.pickle', 'wb') as f:
    pickle.dump(encoder, f)

onehot_Y = to_categorical(labeled_y)  # 레이블을 원-핫 인코딩
print(onehot_Y)  # 원-핫 인코딩 결과 출력

# 형태소 분석 전에 float 값을 문자열로 변환하고, NaN 값을 처리
for i in range(len(X)):
    # float 값을 문자열로 변환하고, NaN 값은 빈 문자열로 대체
    X[i] = str(X[i]) if pd.notna(X[i]) else ''
    X[i] = okt.morphs(X[i], stem=True)  # 형태소 분석 후 어간 추출
print(X)  # 형태소 분석 결과 출력

# 불용어 리스트 읽기
stopwords = pd.read_csv('./crawling_data/stopwords.csv', index_col=0)
print(stopwords)  # 불용어 리스트 출력

# 불용어 제거
for sentence in range(len(X)):
    words = []  # 불용어가 제거된 단어를 저장할 리스트
    for word in range(len(X[sentence])):
        if len(X[sentence][word]) > 1:  # 단어 길이가 1보다 큰 경우
            if X[sentence][word] not in list(stopwords['stopword']):  # 불용어가 아닌 경우
                words.append(X[sentence][word])  # 단어 추가
    X[sentence] = ' '.join(words)  # 단어 리스트를 문자열로 변환하여 저장

print(X[:5])  # 불용어 제거 후 첫 5개 뉴스 제목 출력

token = Tokenizer()  # Tokenizer 객체 생성
token.fit_on_texts(X)  # 텍스트 데이터에 맞춰 토큰화
tokened_X = token.texts_to_sequences(X)  # 텍스트를 시퀀스로 변환
wordsize = len(token.word_index) + 1  # 단어 집합의 크기 계산
print(wordsize)  # 단어 집합의 크기 출력

print(tokened_X[0:5])  # 첫 5개 시퀀스 출력

# # 최대 시퀀스 길이 계산
max = 0
for i in range(len(tokened_X)):
    if max < len(tokened_X[i]):
        max = len(tokened_X[i])  # 최대 시퀀스 길이 업데이트
print(max)  # 최대 시퀀스 길이 출력

with open('./models/MP_KBO_token_max_{}.pickle'.format(max), 'wb') as f:
    pickle.dump(token, f)

X_pad = pad_sequences(tokened_X, max)  # 시퀀스를 패딩하여 동일한 길이로 맞춤
print(X_pad)  # 패딩된 시퀀스 출력
print(len(X_pad[0]))  # 패딩된 시퀀스의 길이 출력

# 데이터 분할
X_train, X_test, Y_train, Y_test = train_test_split(X_pad, onehot_Y, test_size=0.1)  # 10%를 테스트 데이터로 분할
print(X_train.shape, Y_train.shape)  # 학습 데이터의 형태 출력
print(X_test.shape, Y_test.shape)  # 테스트 데이터의 형태 출력

# 데이터 저장
np.save('./crawling_data/MP_KBO_X_train_wordsize_{}_max_{}'.format(wordsize, max), X_train)  # 학습 데이터 저장
np.save('./crawling_data/MP_KBO_Y_train_wordsize_{}_max_{}'.format(wordsize, max), Y_train)  # 학습 레이블 저장
np.save('./crawling_data/MP_KBO_X_test_wordsize_{}_max_{}'.format(wordsize, max), X_test)  # 테스트 데이터 저장
np.save('./crawling_data/MP_KBO_Y_test_wordsize_{}_max_{}'.format(wordsize, max), Y_test)  # 테스트 레이블 저장