import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt


class KBOFanClassifier(QMainWindow):
    def __init__(self):
        super().__init__()

        # Load the UI
        uic.loadUi('./mainWidget.ui', self)

        # Load the model and tokenizer
        self.model = load_model('./models/MP_KBO_category_classification_model_0.5934127569198608.h5')
        with open('models/MP_KBO_token_max_37.pickle', 'rb') as f:
            self.token = pickle.load(f)

        # Load stopwords
        self.stopwords = pd.read_csv('./crawling_data/stopwords.csv', index_col=0)

        # Setup connections
        self.predictButton.clicked.connect(self.predict_fan_type)
        self.actionClear.triggered.connect(self.clear_input)
        self.actionExit.triggered.connect(self.close)

        # Team labels and logo paths
        self.all_labels = ['DOOSAN', 'HANWHA', 'KIA', 'KIWOOM', 'KT', 'LG', 'LOTTE', 'NC', 'SAMSUNG', 'SSG']
        self.team_logos = {
            'DOOSAN': './image/doosan.png',
            'HANWHA': './image/hanwha.png',
            'KIA': './image/kia.png',
            'KIWOOM': './image/kiwoom.png',
            'KT': './image/kt.png',
            'LG': './image/lg.png',
            'LOTTE': './image/lotte.png',
            'NC': './image/nc.png',
            'SAMSUNG': './image/samsung.jpg',
            'SSG': './image/ssg.png'
        }

    def preprocess_text(self, text):
        # Tokenize and clean text
        okt = Okt()
        morphs = okt.morphs(text, stem=True)
        cleaned_words = [word for word in morphs if len(word) > 1 and word not in list(self.stopwords['stopword'])]
        cleaned_text = ' '.join(cleaned_words)

        return cleaned_text

    def predict_fan_type(self):
        # Get input text
        text = self.inputTextEdit.toPlainText()

        if not text:
            QMessageBox.warning(self, "입력 오류", "댓글을 입력해주세요.")
            return

        # Preprocess text
        cleaned_text = self.preprocess_text(text)

        # Tokenize
        tokened_x = self.token.texts_to_sequences([cleaned_text])

        # Pad sequences
        x_pad = pad_sequences(tokened_x, maxlen=37)

        # Predict
        preds = self.model.predict(x_pad)[0]

        # Get top two predictions
        top_two_indices = preds.argsort()[-2:][::-1]
        top_two_teams = [self.all_labels[idx] for idx in top_two_indices]
        top_two_probs = [preds[idx] * 100 for idx in top_two_indices]

        # Update UI with detailed prediction
        prediction_text = (f"1st Most : {top_two_teams[0]} ({top_two_probs[0]:.2f}%)\n"
                           f"2nd Most : {top_two_teams[1]} ({top_two_probs[1]:.2f}%)")

        self.predictionLabel.setText(prediction_text)

        # Update confidence progress bar with top prediction
        self.confidenceProgressBar.setValue(int(top_two_probs[0]))

        # Display team logo
        self.update_team_logo(top_two_teams[0])

    def update_team_logo(self, team):
        # Load and display team logo
        try:
            logo_path = self.team_logos.get(team, '')
            if logo_path:
                pixmap = QPixmap(logo_path)
                scaled_pixmap = pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.teamLogoLabel.setPixmap(scaled_pixmap)
                self.teamLogoLabel.setAlignment(Qt.AlignCenter)
            else:
                self.teamLogoLabel.clear()
        except Exception as e:
            print(f"Error loading logo: {e}")
            self.teamLogoLabel.clear()

    def clear_input(self):
        # Clear input and reset UI
        self.inputTextEdit.clear()
        self.predictionLabel.setText("Predicted Fan Type: -")
        self.confidenceProgressBar.setValue(0)
        self.teamLogoLabel.clear()


def main():
    app = QApplication(sys.argv)
    window = KBOFanClassifier()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()