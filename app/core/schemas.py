"""API response contracts — single source of truth."""
from pydantic import BaseModel
from typing import Optional

class TumorMeta(BaseModel):
    title: str
    who_grade: str
    origin: str
    risk_level: str
    badge_color: str
    description: str

class DifferentialInfo(BaseModel):
    secondary_class: str
    secondary_probability: float
    probability_gap: float
    is_close_call: bool

class PredictionResponse(BaseModel):
    filename: str
    diagnosis: str
    confidence_score: float
    class_probabilities: dict[str, float]
    low_confidence_flag: bool
    tumor_info: Optional[TumorMeta] = None
    differential: Optional[DifferentialInfo] = None
    heatmap_base64: Optional[str] = None
    pure_heatmap_base64: Optional[str] = None