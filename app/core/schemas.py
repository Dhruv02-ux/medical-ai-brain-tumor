"""API response contracts — single source of truth."""
from pydantic import BaseModel

class PredictionResponse(BaseModel):
    filename: str
    diagnosis: str
    confidence_score: float
    class_probabilities: dict[str, float]
    low_confidence_flag: bool