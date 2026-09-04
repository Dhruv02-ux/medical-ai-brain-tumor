"""Grad-CAM heatmap engine — targeting top_activation for accurate spatial explainability."""
import threading
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from app.core.config import GRADCAM_LAYER, IMG_SIZE

_GRAD_MODEL_CACHE: dict[int, tf.keras.Model] = {}
_lock = threading.Lock()

COLORMAPS = {
    "turbo": cv2.COLORMAP_TURBO,
    "jet": cv2.COLORMAP_JET,
    "inferno": cv2.COLORMAP_INFERNO,
    "viridis": cv2.COLORMAP_VIRIDIS,
}

def get_grad_model(model: tf.keras.Model) -> tf.keras.Model:
    """Thread-safe cached Grad-CAM feature-extractor sub-model."""
    model_id = id(model)
    if model_id not in _GRAD_MODEL_CACHE:
        with _lock:
            if model_id not in _GRAD_MODEL_CACHE:
                target_layer = model.get_layer(GRADCAM_LAYER)
                grad_model = tf.keras.Model(
                    inputs=model.inputs,
                    outputs=[target_layer.output, model.output]
                )
                # Warm up
                dummy = np.zeros((1, *IMG_SIZE, 3), dtype=np.float32)
                grad_model(dummy)
                _GRAD_MODEL_CACHE[model_id] = grad_model
    return _GRAD_MODEL_CACHE[model_id]

def generate_heatmap(model: tf.keras.Model, processed_img: np.ndarray, class_idx: int | None = None) -> np.ndarray:
    """
    Computes class-discriminative Grad-CAM localization map.
    If class_idx is None, targets the predicted top-1 class.
    """
    grad_model = get_grad_model(model)
    inputs = tf.cast(processed_img, tf.float32)

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(inputs)
        if class_idx is None:
            class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_output)
    if grads is None:
        return np.zeros((7, 7), dtype=np.float32)

    # Guided alpha-weights: Global-average pool gradients over spatial dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight feature maps by gradient importances
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU: Only features that have a positive influence on the targeted class are preserved
    heatmap = tf.maximum(heatmap, 0.0)
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()

def render_heatmap_colormap(heatmap: np.ndarray, colormap_name: str = "turbo") -> np.ndarray:
    """Converts 2D normalized heatmap to an RGB colormap image matching IMG_SIZE."""
    heatmap_resized = cv2.resize(heatmap, IMG_SIZE, interpolation=cv2.INTER_CUBIC)
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap_resized, 0.0, 1.0))
    cmap = COLORMAPS.get(colormap_name.lower(), cv2.COLORMAP_TURBO)
    colored_bgr = cv2.applyColorMap(heatmap_uint8, cmap)
    return cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)

def overlay_heatmap_on_image(
    heatmap: np.ndarray,
    original_image: Image.Image,
    alpha: float = 0.55,
    colormap_name: str = "turbo",
    threshold: float = 0.10
) -> np.ndarray:
    """
    Blends Grad-CAM activation seamlessly onto the original MRI scan.
    Uses intensity-weighted alpha masking: regions with low/zero activation (< threshold)
    remain 100% transparent and crystal clear, eliminating annoying dark blue background wash.
    """
    original = np.array(original_image.convert("RGB").resize(IMG_SIZE, Image.Resampling.BILINEAR), dtype=np.uint8)
    heatmap_resized = cv2.resize(heatmap, IMG_SIZE, interpolation=cv2.INTER_CUBIC)
    heatmap_resized = np.clip(heatmap_resized, 0.0, 1.0)
    
    # Generate colormap
    colored_rgb = render_heatmap_colormap(heatmap, colormap_name=colormap_name)

    # Intensity weight: smooth transition above threshold
    mask = np.clip((heatmap_resized - threshold) / (1.0 - threshold + 1e-6), 0.0, 1.0)
    effective_alpha = (mask * alpha)[..., np.newaxis]

    blended = (original * (1.0 - effective_alpha) + colored_rgb * effective_alpha).astype(np.uint8)
    return blended