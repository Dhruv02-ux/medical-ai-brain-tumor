"""Single responsibility: image in -> diagnosis dict out."""
import numpy as np
from PIL import Image
from app.core.model_loader import get_model
from app.core.preprocessing import preprocess_mri
from app.core.config import CLASSES, LOW_CONFIDENCE_THRESHOLD

def predict_tumor(image: Image.Image) -> dict:
    model = get_model()
    probabilities = model.predict(preprocess_mri(image), verbose=0)[0]
    top_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[top_idx])

    return {
        "diagnosis": CLASSES[top_idx],
        "confidence_score": round(confidence, 4),
        "class_probabilities": {CLASSES[i]: round(float(p), 4) for i, p in enumerate(probabilities)},
        "low_confidence_flag": confidence < LOW_CONFIDENCE_THRESHOLD,
    }