"""Single responsibility: image in -> diagnosis dict out."""
import numpy as np
from PIL import Image
from app.core.model_loader import get_model
from app.core.preprocessing import preprocess_mri
from app.core.config import CLASSES, LOW_CONFIDENCE_THRESHOLD, DIFFERENTIAL_GAP_THRESHOLD, TUMOR_INFO

def predict_tumor(image: Image.Image) -> dict:
    model = get_model()
    processed = preprocess_mri(image)
    probabilities = model(processed, training=False).numpy()[0]
    
    top_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[top_idx])
    diagnosis = CLASSES[top_idx]

    # Differential gap analysis
    sorted_indices = np.argsort(probabilities)[::-1]
    second_idx = int(sorted_indices[1])
    second_class = CLASSES[second_idx]
    second_prob = float(probabilities[second_idx])
    gap = float(confidence - second_prob)

    differential = {
        "secondary_class": second_class,
        "secondary_probability": round(second_prob, 4),
        "probability_gap": round(gap, 4),
        "is_close_call": gap < DIFFERENTIAL_GAP_THRESHOLD,
    }

    meta = TUMOR_INFO.get(diagnosis, {
        "title": diagnosis.capitalize(),
        "who_grade": "Clinical Review Needed",
        "origin": "Parenchyma / Meninges",
        "risk_level": "Under Evaluation",
        "badge_color": "slate",
        "description": "Correlation with formal radiology reading and histopathology recommended."
    })

    return {
        "diagnosis": diagnosis,
        "confidence_score": round(confidence, 4),
        "class_probabilities": {CLASSES[i]: round(float(p), 4) for i, p in enumerate(probabilities)},
        "low_confidence_flag": confidence < LOW_CONFIDENCE_THRESHOLD,
        "tumor_info": meta,
        "differential": differential,
    }