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
CLASSES: list[str] = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE: tuple[int, int] = (224, 224)
GRADCAM_LAYER: str = "top_activation"  # verified from actual model.keras layer inspection

LOW_CONFIDENCE_THRESHOLD: float = 0.70
DIFFERENTIAL_GAP_THRESHOLD: float = 0.20

LLM_MODEL: str = "Qwen/Qwen2.5-7B-Instruct"
HF_ROUTER_URL: str = "https://router.huggingface.co/v1"
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"  # called remotely, not loaded locally
VECTORSTORE_PATH: str = "vectorstore/who_cns_index"
MAX_TOKENS_REPORT: int = 400
LLM_TIMEOUT_SECONDS: int = 15