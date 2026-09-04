import io
import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError
from app.core.config import IMG_SIZE

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg"}
MIN_DIMENSION = 50

def validate_and_load_image(content_type: str | None, image_bytes: bytes) -> Image.Image:
    if not image_bytes:
        raise ValueError("Empty file received. Please provide a valid brain MRI scan.")
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
        image = image.convert("RGB")
    except (UnidentifiedImageError, Exception):
        raise ValueError("File is not a valid or readable image. Please provide a standard image (JPEG, PNG, WEBP).")
    if min(image.size) < MIN_DIMENSION:
        raise ValueError(f"Image too small ({image.size[0]}x{image.size[1]}px); minimum {MIN_DIMENSION}px per side.")
    return image


def preprocess_mri(image: Image.Image) -> np.ndarray:
    """
    Standardize MRI scan for EfficientNetB0 inference.
    
    NOTE: The trained EfficientNetB0 architecture includes an internal
    Rescaling(1./255) layer. Therefore, the input MUST remain in the range [0.0, 255.0].
    Dividing by 255 here causes double-normalization and catastrophic model collapse.
    
    Also, direct resize ensures 1:1 spatial coordinate alignment with Grad-CAM heatmaps.
    """
    resized = image.convert("RGB").resize(IMG_SIZE, Image.Resampling.BILINEAR)
    img_array = np.array(resized, dtype=np.float32)
    return np.expand_dims(img_array, axis=0)