import sys
import pandas as pd  # CSV 파일을 읽기 위해 pandas를 사용합니다.
from PyQt5.QtWidgets import *
from PyQt5 import uic
from keras.models import load_model

form_window = uic.loadUiType('./mainWidget.ui')[0]

class Exam(QWidget, form_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.model = load_model(
            './models/MP_KBO_category_classification_model_0.6137456893920898.h5')
        self.btn_save.clicked.connect(self.save_and_predict)  # CSV 저장 및 예측 버튼 연결

    def save_and_predict(self):
        text = self.textEdit_repl.toPlainText()  # QTextEdit에서 텍스트 가져오기
        if text:  # 텍스트가 비어있지 않은 경우
            # CSV 파일로 저장
            with open('./crawling_data/output.csv', 'w', encoding='utf-8') as file:  # CSV 파일 열기
                file.write('titles,category\n')
                file.write(text)  # 텍스트를 파일에 쓰기
                file.write(',KBO')
            print("CSV 파일로 저장되었습니다.")
            self.lbl_predict.setText("CSV 파일로 저장되었습니다.")

            # CSV 파일 읽기
            df = pd.read_csv('./crawling_data/output.csv')
            # 예측을 위한 데이터 전처리 (여기서는 예시로 첫 번째 행을 사용)
            if not df.empty:
                input_data = df.iloc[0].values.reshape(1, -1)  # 첫 번째 행을 2D 배열로 변환
                prediction = self.model.predict(input_data)  # 모델 예측
                print(prediction)

                # 예측 결과를 레이블에 표시 (예시로 0.5를 기준으로 팀을 예측)
                if prediction < 0.5:
                    self.lbl_predict.setText("예측 결과: 팀 A")
                    print(prediction)
                else:
                    self.lbl_predict.setText("예측 결과: 팀 B")
                    print(prediction)
        else:
            self.lbl_predict.setText("저장할 텍스트가 없습니다.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Exam()
    window.show()
    sys.exit(app.exec_())