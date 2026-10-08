import os
import sys

# Ensure root directory is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# File Paths
DATA_DIR = os.path.join(BASE_DIR, "data")
ARCHIVE_ZIP = os.path.join(BASE_DIR, "archive.zip")
RAW_DATA_PATH = os.path.join(DATA_DIR, "train.csv")
CLEANED_DATA_PATH = os.path.join(DATA_DIR, "cleaned_hatespeech.csv")

MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "hatespeech_model.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")

STATIC_DIR = os.path.join(BASE_DIR, "static")
IMAGES_DIR = os.path.join(STATIC_DIR, "images")
WEB_DIR = os.path.join(BASE_DIR, "web")

# Ensure required directories exist
for directory in [DATA_DIR, MODELS_DIR, STATIC_DIR, IMAGES_DIR, WEB_DIR]:
    os.makedirs(directory, exist_ok=True)

# Class Label Mapping
CLASS_MAP = {
    0: "Hate Speech",
    1: "Offensive Language",
    2: "Neither"
}

CLASS_DESCRIPTIONS = {
    0: "Contains hate speech, racial slurs, or incitement to hatred against protected groups.",
    1: "Contains profane or offensive language without direct targeted hate speech.",
    2: "Neutral, clean, or normal everyday language."
}

# Feature Extraction Settings
MAX_FEATURES = 10000
NGRAM_RANGE = (1, 2)

# Random Seed for Reproducibility
RANDOM_STATE = 42
