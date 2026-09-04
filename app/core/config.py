import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN: str = os.getenv("HF_TOKEN", "")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN missing. Set it in .env (local) or Render env vars (prod).")

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY missing. Set it in .env (local) or Render env vars (prod).")

MODEL_PATH: str = "models/brain_tumor_model.keras"
# Strictly match training flow_from_directory alphabetical ordering:
# 0: glioma, 1: meningioma, 2: notumor, 3: pituitary
CLASSES: list[str] = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE: tuple[int, int] = (224, 224)

# Target top_activation (activated 2D feature map of shape (7, 7, 1280))
GRADCAM_LAYER: str = "top_activation"

LOW_CONFIDENCE_THRESHOLD: float = 0.70
DIFFERENTIAL_GAP_THRESHOLD: float = 0.20

TUMOR_INFO: dict[str, dict[str, str]] = {
    "glioma": {
        "title": "Glioma",
        "who_grade": "WHO Grade II - IV",
        "origin": "Glial cells (Astrocytes / Oligodendrocytes)",
        "risk_level": "High Risk",
        "badge_color": "rose",
        "description": "Intra-axial tumor arising from neuroglial progenitor cells with infiltrative growth along white matter tracts.",
    },
    "meningioma": {
        "title": "Meningioma",
        "who_grade": "WHO Grade I - III",
        "origin": "Arachnoid cap cells of meninges",
        "risk_level": "Moderate Risk",
        "badge_color": "amber",
        "description": "Predominantly extra-axial mass arising from meningothelial cells, often showing dural tail enhancement.",
    },
    "pituitary": {
        "title": "Pituitary Tumor (PitNET)",
        "who_grade": "WHO Grade I Neuroendocrine",
        "origin": "Anterior pituitary fossa",
        "risk_level": "Moderate Risk",
        "badge_color": "indigo",
        "description": "Sellar mass originating from adenohypophyseal cells, frequently presenting with optic chiasm compression or hormonal changes.",
    },
    "notumor": {
        "title": "No Neoplasm Detected",
        "who_grade": "Normal Brain MRI",
        "origin": "Preserved parenchyma",
        "risk_level": "Normal",
        "badge_color": "emerald",
        "description": "Symmetrical cerebral architecture with no focal mass lesions, pathological enhancement, or ventricular distortion.",
    },
}

LLM_MODEL: str = "Qwen/Qwen2.5-7B-Instruct"
HF_ROUTER_URL: str = "https://router.huggingface.co/v1"
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"  # called remotely, not loaded locally
VECTORSTORE_PATH: str = "vectorstore/who_cns_index"
MAX_TOKENS_REPORT: int = 400
LLM_TIMEOUT_SECONDS: int = 15