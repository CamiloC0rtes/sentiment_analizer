"""
Sentiment Analysis Model Training Script using IMDb Dataset from Kaggle
Trains a sentiment classifier and saves it as a pickle file
"""

import pandas as pd
import numpy as np
import joblib
import re
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import warnings
warnings.filterwarnings('ignore')

# Descargar recursos de NLTK
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

class SentimentModel:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.model = None
        
    def preprocess_text(self, text):
        """Limpia y preprocesa el texto"""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
        text = re.sub(r'\@w+|\#','', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        words = text.split()
        words = [self.stemmer.stem(word) for word in words if word not in self.stop_words]
        return ' '.join(words)
    
    def load_data_from_imdb(self, path='IMDB Dataset.csv'):
        """Carga el dataset de IMDb desde un archivo CSV"""
        print(f"Cargando datos desde {path}...")
        df = pd.read_csv(path)
        df = df[['review', 'sentiment']]
        df['sentiment'] = df['sentiment'].map({'negative': 0, 'positive': 2})  # 0 = negativo, 2 = positivo
        return df

    def train_model(self, df):
        """Entrena el modelo de análisis de sentimiento"""
        print(f"Entrenando modelo con {len(df)} muestras...")
        df['processed_text'] = df['review'].apply(self.preprocess_text)
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'], 
            df['sentiment'], 
            test_size=0.2, 
            random_state=42,
            stratify=df['sentiment']
        )
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                stop_words='english'
            )),
            ('classifier', MultinomialNB(alpha=1.0))
        ])
        print("Entrenando modelo...")
        self.model.fit(X_train, y_train)
        print(f"Precisión en entrenamiento: {self.model.score(X_train, y_train):.3f}")
        print(f"Precisión en prueba: {self.model.score(X_test, y_test):.3f}")
        print("Validación cruzada...")
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
        print(f"Accuracy CV: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        y_pred = self.model.predict(X_test)
        print("\nReporte de Clasificación:")
        print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
        return self.model
    
    def save_model(self, filepath='model/model.pkl'):
        """Guarda el modelo entrenado"""
        if self.model is None:
            raise ValueError("No hay modelo entrenado para guardar.")
        joblib.dump({
            'model': self.model,
            'stemmer': self.stemmer,
            'stop_words': self.stop_words
        }, filepath)
        print(f"Modelo guardado en {filepath}")
    
    def predict(self, text):
        """Predice el sentimiento de un texto dado"""
        if self.model is None:
            raise ValueError("Modelo no entrenado.")
        processed_text = self.preprocess_text(text)
        prediction = self.model.predict([processed_text])[0]
        confidence = self.model.predict_proba([processed_text])[0]
        sentiment_map = {0: 'negative', 2: 'positive'}
        return {
            'sentiment': sentiment_map[prediction],
            'confidence': float(max(confidence)),
            'scores': {
                'negative': float(confidence[0]),
                'positive': float(confidence[1])
            }
        }

def main():
    print("Iniciando entrenamiento del modelo de sentimiento (IMDb)...")
    model = SentimentModel()
    df = model.load_data_from_imdb('IMDB Dataset.csv')  # Asegúrate que el archivo esté en esta ruta
    model.train_model(df)
    model.save_model()
    
    # Prueba de predicción
    print("\nEjemplo de predicción:")
    test_texts = [
        "I absolutely loved this movie! A masterpiece.",
        "This was terrible, I want my time back."
    ]
    for text in test_texts:
        result = model.predict(text)
        print(f"Texto: {text}")
        print(f"Sentimiento: {result['sentiment']} (confianza: {result['confidence']:.3f})\n")

if __name__ == '__main__':
    main()
