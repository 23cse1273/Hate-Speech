import os
import sys
import re
import zipfile

# Ensure root directory is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from src.config import RAW_DATA_PATH, CLEANED_DATA_PATH, ARCHIVE_ZIP, DATA_DIR

# Ensure NLTK resources are available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

class TextPreprocessor:
    """
    Handles data loading, text cleaning, normalization, stemming, 
    and generating summary reports for data preprocessing.
    """
    def __init__(self):
        self.stemmer = PorterStemmer()
        try:
            self.stop_words = set(stopwords.words('english'))
            # Keep certain negation words as they affect sentiment
            self.stop_words -= {'not', 'no', 'nor', 'never'}
        except Exception:
            self.stop_words = set()

    def clean_text(self, text: str) -> str:
        """
        Cleans raw input text:
        1. Converts to lowercase.
        2. Removes URLs, Twitter handles (@user), RT tags, HTML entities.
        3. Removes non-alphabetic characters (digits, special punctuation).
        4. Removes stop words and applies stemming.
        5. Normalizes extra whitespace.
        """
        if not isinstance(text, str):
            return ""
        
        # 1. Lowercase
        text = text.lower()
        
        # 2. Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # 3. Remove HTML tags & entities
        text = re.sub(r'&amp;|&lt;|&gt;', '', text)
        
        # 4. Remove Twitter handles (@username) and RT tags
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'\brt\b', '', text)
        
        # 5. Remove numbers and non-alpha characters
        text = re.sub(r'[^a-z\s]', ' ', text)
        
        # 6. Tokenize, remove stopwords, and stem
        tokens = text.split()
        cleaned_tokens = [
            self.stemmer.stem(word) 
            for word in tokens 
            if word not in self.stop_words and len(word) > 1
        ]
        
        return " ".join(cleaned_tokens)

    def load_raw_data(self) -> pd.DataFrame:
        """
        Loads raw dataset from CSV file or extracts it from archive.zip if needed.
        """
        if not os.path.exists(RAW_DATA_PATH):
            if os.path.exists(ARCHIVE_ZIP):
                os.makedirs(DATA_DIR, exist_ok=True)
                with zipfile.ZipFile(ARCHIVE_ZIP, 'r') as zip_ref:
                    zip_ref.extract('train.csv', DATA_DIR)
                print(f"[INFO] Extracted raw dataset from {ARCHIVE_ZIP} to {RAW_DATA_PATH}")
            else:
                raise FileNotFoundError(f"Raw data not found at {RAW_DATA_PATH} or {ARCHIVE_ZIP}")
        
        df = pd.read_csv(RAW_DATA_PATH)
        return df

    def preprocess_dataset(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Preprocesses raw dataset and saves cleaned version to CSV.
        """
        if df is None:
            df = self.load_raw_data()
            
        print("[INFO] Cleaning text dataset...")
        df['cleaned_tweet'] = df['tweet'].apply(self.clean_text)
        
        # Calculate text length metrics
        df['raw_length'] = df['tweet'].astype(str).apply(len)
        df['cleaned_length'] = df['cleaned_tweet'].astype(str).apply(len)
        df['raw_word_count'] = df['tweet'].astype(str).apply(lambda x: len(x.split()))
        df['cleaned_word_count'] = df['cleaned_tweet'].astype(str).apply(lambda x: len(x.split()))

        # Remove empty rows after cleaning if any
        df = df[df['cleaned_tweet'].str.strip() != ''].copy()
        
        df.to_csv(CLEANED_DATA_PATH, index=False)
        print(f"[SUCCESS] Cleaned dataset saved to {CLEANED_DATA_PATH} ({len(df)} rows)")
        return df

    def generate_preprocessing_report(self, df: pd.DataFrame = None) -> str:
        """
        Generates a detailed text-formatted conclusion report summarizing
        all data preprocessing steps, metrics, and insights.
        """
        if df is None:
            if os.path.exists(CLEANED_DATA_PATH):
                df = pd.read_csv(CLEANED_DATA_PATH)
            else:
                df = self.preprocess_dataset()

        raw_avg_len = df['raw_length'].mean()
        clean_avg_len = df['cleaned_length'].mean()
        raw_avg_words = df['raw_word_count'].mean()
        clean_avg_words = df['cleaned_word_count'].mean()
        
        raw_vocab = set(" ".join(df['tweet'].astype(str)).lower().split())
        clean_vocab = set(" ".join(df['cleaned_tweet'].astype(str)).split())
        
        vocab_reduction = ((len(raw_vocab) - len(clean_vocab)) / len(raw_vocab)) * 100
        length_reduction = ((raw_avg_len - clean_avg_len) / raw_avg_len) * 100

        report = f"""================================================================================
                    DATA PREPROCESSING CONCLUSION & SUMMARY REPORT
================================================================================
1. OVERVIEW & OBJECTIVES:
   - Total Raw Records Processed : {len(df):,}
   - Primary Target Feature       : 'tweet' -> 'cleaned_tweet'
   - Target Classes               : Class 0 (Hate Speech), Class 1 (Offensive), Class 2 (Neither)

2. STEP-BY-STEP PREPROCESSING PIPELINE:
   [Step 1] Text Lowercasing: Standardized all text to lowercase to treat capitalized words uniformly.
   [Step 2] Noise Removal    : Stripped URLs, @usernames, 'RT' retweet markers, HTML entities (&amp;), digits, and special characters.
   [Step 3] Stopwords Removal: Eliminated high-frequency non-informative English stop words (retaining key negations).
   [Step 4] Stemming         : Applied NLTK PorterStemmer to reduce words to their root form (e.g., 'running' -> 'run').
   [Step 5] Whitespace Fix   : Collapsed multiple spaces and stripped leading/trailing spaces.

3. QUANTITATIVE REDUCTION METRICS:
   - Average Character Length (Before Cleaning) : {raw_avg_len:.2f} characters
   - Average Character Length (After Cleaning)  : {clean_avg_len:.2f} characters
   - Character Length Reduction Ratio           : {length_reduction:.2f}%
   
   - Average Word Count (Before Cleaning)       : {raw_avg_words:.2f} words
   - Average Word Count (After Cleaning)        : {clean_avg_words:.2f} words
   
   - Raw Total Vocabulary Size                  : {len(raw_vocab):,} unique tokens
   - Cleaned Total Vocabulary Size              : {len(clean_vocab):,} unique tokens
   - Vocabulary Complexity Reduction            : {vocab_reduction:.2f}%

4. KEY PREPROCESSING CONCLUSION:
   - Noise Elimination: Successfully removed social media artifacts (hashtags, handles, URLs, RTs) which lack semantic value for hate speech detection.
   - Dimensionality Reduction: The preprocessing pipeline reduced total unique vocabulary size by {vocab_reduction:.2f}%, significantly reducing vector space dimensionality while concentrating signal density.
   - Feature Quality: Stemming unified morphological variants of offensive terms, allowing the classifier to generalize better across unseen user posts.
================================================================================
"""
        return report

if __name__ == "__main__":
    preprocessor = TextPreprocessor()
    df = preprocessor.preprocess_dataset()
    report = preprocessor.generate_preprocessing_report(df)
    print(report)
