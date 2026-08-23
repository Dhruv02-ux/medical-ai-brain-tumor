"""Input validation + preprocessing — nothing bad reaches the model."""
import io
import numpy as np
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.applications.efficientnet import preprocess_input
from app.core.config import IMG_SIZE

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg"}
MIN_DIMENSION = 50

def validate_and_load_image(content_type: str, image_bytes: bytes) -> Image.Image:
    if content_type not in ALLOWED_TYPES:
        raise ValueError(f"Unsupported file type: {content_type}. Use JPEG or PNG.")
    if not image_bytes:
        raise ValueError("Empty file received.")
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError:
        raise ValueError("File is not a valid image.")
    if min(image.size) < MIN_DIMENSION:
        raise ValueError(f"Image too small {image.size}; minimum {MIN_DIMENSION}px per side.")
    return image

def preprocess_mri(image: Image.Image) -> np.ndarray:
    resized = image.resize(IMG_SIZE)
    img_array = np.expand_dims(np.array(resized, dtype=np.float32), axis=0)
    return preprocess_input(img_array)