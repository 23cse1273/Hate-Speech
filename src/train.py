import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

# Ensure root directory is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.config import (
    CLEANED_DATA_PATH, MODEL_PATH, VECTORIZER_PATH, METRICS_PATH, 
    RANDOM_STATE, CLASS_MAP
)
from src.preprocessing import TextPreprocessor
from src.feature_extraction import FeatureExtractor
from src.visualize import DataVisualizer

class ModelTrainer:
    """
    Trains multiple ML classifiers, evaluates metrics, selects the best model,
    saves artifacts, and exports confusion matrix plots & metrics JSON.
    """
    def __init__(self, data_path: str = CLEANED_DATA_PATH):
        if not os.path.exists(data_path):
            preprocessor = TextPreprocessor()
            self.df = preprocessor.preprocess_dataset()
        else:
            self.df = pd.read_csv(data_path)

        # Drop any NaN values if present
        self.df = self.df.dropna(subset=['cleaned_tweet']).copy()

    def train_and_evaluate(self):
        print("================================================================================")
        print("                   MACHINE LEARNING TRAINING PIPELINE                          ")
        print("================================================================================")

        X_text = self.df['cleaned_tweet'].astype(str).values
        y = self.df['class'].values

        # 1. Train-Test Split (80% Train, 20% Test, Stratified)
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            X_text, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
        )
        print(f"[INFO] Train set size: {len(X_train_text):,} | Test set size: {len(X_test_text):,}")

        # 2. Vectorize Features
        extractor = FeatureExtractor()
        X_train = extractor.fit_transform(X_train_text)
        X_test = extractor.transform(X_test_text)
        extractor.save_vectorizer(VECTORIZER_PATH)

        # 3. Define Candidate Models
        models = {
            "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE),
            "Linear Support Vector Classifier (LinearSVC)": LinearSVC(class_weight='balanced', max_iter=2000, random_state=RANDOM_STATE),
            "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
            "Stochastic Gradient Descent (SGD)": SGDClassifier(loss='log_loss', class_weight='balanced', random_state=RANDOM_STATE)
        }

        results = {}
        best_model_name = None
        best_macro_f1 = -1.0
        best_model_obj = None

        print("\n--- Model Comparison Benchmark ---")
        for name, clf in models.items():
            print(f"Training {name}...")
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
            w_prec, w_rec, w_f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

            results[name] = {
                "accuracy": round(float(acc), 4),
                "macro_precision": round(float(precision), 4),
                "macro_recall": round(float(recall), 4),
                "macro_f1": round(float(f1), 4),
                "weighted_f1": round(float(w_f1), 4)
            }

            print(f"  -> Accuracy: {acc:.4f} | Macro F1: {f1:.4f} | Weighted F1: {w_f1:.4f}")

            if f1 > best_macro_f1:
                best_macro_f1 = f1
                best_model_name = name
                best_model_obj = clf

        print("\n================================================================================")
        print(f"[BEST MODEL] Winner: {best_model_name} (Macro F1-Score: {best_macro_f1:.4f})")
        print("================================================================================")

        # 4. Detailed Evaluation of Winner
        y_best_pred = best_model_obj.predict(X_test)
        report_str = classification_report(
            y_test, y_best_pred, 
            target_names=[CLASS_MAP[0], CLASS_MAP[1], CLASS_MAP[2]]
        )
        print("\n--- Detailed Classification Report for Best Model ---")
        print(report_str)

        cm = confusion_matrix(y_test, y_best_pred)
        visualizer = DataVisualizer()
        visualizer.plot_confusion_matrix(cm)

        # 5. Save Model Artifacts
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(best_model_obj, MODEL_PATH)
        print(f"[SUCCESS] Trained model ({best_model_name}) saved to {MODEL_PATH}")

        # Save Metrics JSON
        metrics_payload = {
            "best_model": best_model_name,
            "best_macro_f1": round(float(best_macro_f1), 4),
            "all_models": results,
            "confusion_matrix": cm.tolist(),
            "target_classes": CLASS_MAP
        }
        with open(METRICS_PATH, 'w') as f:
            json.dump(metrics_payload, f, indent=2)
        print(f"[SUCCESS] Evaluation metrics exported to {METRICS_PATH}")

        return best_model_obj, results

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train_and_evaluate()
