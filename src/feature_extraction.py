import os
import sys
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Ensure root directory is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.config import VECTORIZER_PATH, MAX_FEATURES, NGRAM_RANGE

class FeatureExtractor:
    """
    Handles TF-IDF feature extraction, vectorizer fitting, saving, and loading.
    """
    def __init__(self, max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            min_df=2
        )

    def fit_transform(self, corpus: list):
        """
        Fits vectorizer on text corpus and returns TF-IDF matrix.
        """
        print(f"[INFO] Fitting TF-IDF Vectorizer (max_features={self.vectorizer.max_features}, ngrams={self.vectorizer.ngram_range})...")
        X = self.vectorizer.fit_transform(corpus)
        print(f"[SUCCESS] Feature matrix shape: {X.shape}")
        return X

    def transform(self, corpus: list):
        """
        Transforms text corpus using fitted vectorizer.
        """
        return self.vectorizer.transform(corpus)

    def save_vectorizer(self, filepath: str = VECTORIZER_PATH):
        """
        Saves vectorizer instance to disk.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.vectorizer, filepath)
        print(f"[SUCCESS] TF-IDF Vectorizer saved to {filepath}")

    def load_vectorizer(self, filepath: str = VECTORIZER_PATH):
        """
        Loads vectorizer instance from disk.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Vectorizer file not found at {filepath}")
        self.vectorizer = joblib.load(filepath)
        print(f"[SUCCESS] TF-IDF Vectorizer loaded from {filepath}")
        return self.vectorizer

    def get_top_ngrams(self, corpus: list, n: int = 10, ngram_range=(1, 1)) -> list:
        """
        Returns top N most frequent n-grams in the corpus.
        """
        vec = CountVectorizer(ngram_range=ngram_range, stop_words='english').fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        return words_freq[:n]

if __name__ == "__main__":
    from src.preprocessing import TextPreprocessor
    preprocessor = TextPreprocessor()
    df = preprocessor.preprocess_dataset()
    
    extractor = FeatureExtractor()
    X = extractor.fit_transform(df['cleaned_tweet'])
    extractor.save_vectorizer()
    
    top_unigrams = extractor.get_top_ngrams(df['cleaned_tweet'], n=5, ngram_range=(1,1))
    print(f"\nTop 5 Unigrams across corpus: {top_unigrams}")
