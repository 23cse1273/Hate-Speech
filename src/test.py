import os
import sys
import argparse
import joblib
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Ensure root directory is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.config import MODEL_PATH, VECTORIZER_PATH, CLEANED_DATA_PATH, CLASS_MAP, CLASS_DESCRIPTIONS
from src.preprocessing import TextPreprocessor

class ModelTester:
    """
    Standalone testing script for developers to evaluate trained models
    and test custom sample inputs.
    """
    def __init__(self, model_path=MODEL_PATH, vectorizer_path=VECTORIZER_PATH):
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            raise FileNotFoundError("Model or Vectorizer not found. Please run 'python src/train.py' first.")
        
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.preprocessor = TextPreprocessor()

    def predict_sample(self, raw_text: str) -> dict:
        """
        Runs full inference pipeline on a single raw text string.
        """
        cleaned_text = self.preprocessor.clean_text(raw_text)
        if not cleaned_text:
            return {
                "raw_text": raw_text,
                "cleaned_text": "",
                "class_id": 2,
                "class_name": CLASS_MAP[2],
                "confidence": 1.0,
                "probabilities": {CLASS_MAP[i]: (1.0 if i == 2 else 0.0) for i in range(3)},
                "description": CLASS_DESCRIPTIONS[2]
            }

        X_vec = self.vectorizer.transform([cleaned_text])
        pred_class = int(self.model.predict(X_vec)[0])

        # Get probabilities or decision function scores
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_vec)[0]
        else:
            dec_scores = self.model.decision_function(X_vec)[0]
            exp_scores = np.exp(dec_scores - np.max(dec_scores))
            probs = exp_scores / exp_scores.sum()

        prob_dict = {CLASS_MAP[i]: round(float(probs[i]), 4) for i in range(3)}
        confidence = round(float(probs[pred_class]), 4)

        return {
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "class_id": pred_class,
            "class_name": CLASS_MAP[pred_class],
            "confidence": confidence,
            "probabilities": prob_dict,
            "description": CLASS_DESCRIPTIONS[pred_class]
        }

    def evaluate_on_file(self, data_path=CLEANED_DATA_PATH):
        """
        Evaluates saved model performance on cleaned dataset.
        """
        print(f"[INFO] Evaluating model on {data_path}...")
        df = pd.read_csv(data_path).dropna(subset=['cleaned_tweet'])
        X_vec = self.vectorizer.transform(df['cleaned_tweet'].astype(str))
        y_true = df['class'].values
        y_pred = self.model.predict(X_vec)

        acc = accuracy_score(y_true, y_pred)
        print(f"\nOverall Dataset Accuracy: {acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=[CLASS_MAP[i] for i in range(3)]))
        return acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hate Speech ML Model Testing & Inference Tool")
    parser.add_argument("--text", type=str, help="Custom text string to classify")
    parser.add_argument("--eval", action="store_true", help="Run model evaluation on dataset")
    args = parser.parse_args()

    tester = ModelTester()

    if args.text:
        res = tester.predict_sample(args.text)
        print("\n==================================================")
        print("                INFERENCE RESULT                  ")
        print("==================================================")
        print(f"Raw Input    : {res['raw_text']}")
        print(f"Cleaned Text : {res['cleaned_text']}")
        print(f"Prediction   : {res['class_name']} (Class {res['class_id']})")
        print(f"Confidence   : {res['confidence'] * 100:.1f}%")
        print("Class Probabilities:")
        for cls_name, prob in res['probabilities'].items():
            print(f"  - {cls_name:<20}: {prob*100:.1f}%")
        print(f"Description  : {res['description']}")
        print("==================================================\n")
    elif args.eval:
        tester.evaluate_on_file()
    else:
        # Default test samples for developers
        sample_texts = [
            "Good morning everyone! Hope you all have a wonderful day ahead.",
            "Shut up you stupid b***h, nobody asked for your opinion!",
            "I hate all [slur] people, they should be eliminated from our country."
        ]
        print("\n[INFO] No CLI args provided. Running test samples...\n")
        for sample in sample_texts:
            res = tester.predict_sample(sample)
            print(f"Input: \"{sample}\"")
            print(f"  -> Prediction: {res['class_name']} (Conf: {res['confidence']*100:.1f}%)")
            print(f"  -> Probabilities: {res['probabilities']}\n")
