import os
import joblib
from pathlib import Path
from typing import Dict, Any
from app.config import MODEL_PATH
from app.ml.features import LinguisticFeatureExtractor
from app.ml.trainer import train_and_save_model

_cached_model = None

def get_model():
    """Returns the loaded classifier pipeline, training if necessary."""
    global _cached_model
    if _cached_model is not None:
        return _cached_model

    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found at {MODEL_PATH}. Initiating training...")
        _cached_model = train_and_save_model(MODEL_PATH)
    else:
        try:
            _cached_model = joblib.load(MODEL_PATH)
        except Exception as e:
            print(f"Failed to load cached model ({e}). Re-training...")
            _cached_model = train_and_save_model(MODEL_PATH)

    return _cached_model

def predict_text_credibility(text: str) -> Dict[str, Any]:
    """
    Performs ML inference on the submitted text.
    Returns:
    - predicted_label: 'REAL' or 'FAKE'
    - fake_probability: float 0.0 to 100.0
    - real_probability: float 0.0 to 100.0
    - model_confidence: float 0.0 to 100.0
    - linguistic_markers: Dict detailing sensationalism, punctuation, capitalization
    """
    if not text or not text.strip():
        return {
            "predicted_label": "UNABLE TO VERIFY",
            "fake_probability": 50.0,
            "real_probability": 50.0,
            "model_confidence": 0.0,
            "linguistic_markers": {}
        }

    pipeline = get_model()
    
    probs = pipeline.predict_proba([text])[0]
    p_real = float(probs[0]) * 100.0
    p_fake = float(probs[1]) * 100.0

    model_confidence = max(p_real, p_fake)
    predicted_label = "FAKE" if p_fake > p_real else "REAL"

    extractor = LinguisticFeatureExtractor()
    features = extractor.transform([text])[0]
    
    markers = {
        "caps_ratio": round(float(features[0]) * 100, 1),
        "exclamations_per_100w": round(float(features[1]) * 100, 1),
        "question_marks": round(float(features[2]) * 100, 1),
        "sensational_score": round(float(features[4]) * 10, 1),
        "journalistic_attribution": round(float(features[5]) * 10, 1),
        "num_data_tokens": int(features[6] * len(text.split()))
    }

    return {
        "predicted_label": predicted_label,
        "fake_probability": round(p_fake, 1),
        "real_probability": round(p_real, 1),
        "model_confidence": round(model_confidence, 1),
        "linguistic_markers": markers
    }
