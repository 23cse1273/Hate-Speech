import os
import sys

# Ensure root directory is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.test import ModelTester

# Global cached instance for fast API serving
_tester_instance = None

def get_tester():
    global _tester_instance
    if _tester_instance is None:
        _tester_instance = ModelTester()
    return _tester_instance

def predict_text(raw_text: str) -> dict:
    """
    Public inference function. Given raw text, returns classification dictionary.
    """
    tester = get_tester()
    return tester.predict_sample(raw_text)

if __name__ == "__main__":
    result = predict_text("Have a great day ahead!")
    print(result)
