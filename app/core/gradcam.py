"""Grad-CAM heatmap — uses verified 'top_activation' layer from the trained model."""
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from app.core.config import GRADCAM_LAYER, IMG_SIZE

def generate_heatmap(model: tf.keras.Model, processed_img: np.ndarray) -> np.ndarray:
    grad_model = tf.keras.Model([model.inputs], [model.get_layer(GRADCAM_LAYER).output, model.output])
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(processed_img)
        loss = predictions[:, tf.argmax(predictions[0])]
    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_mean(conv_output[0] * pooled_grads, axis=-1)
    return (tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)).numpy()

def overlay_heatmap_on_image(heatmap: np.ndarray, original_image: Image.Image, alpha: float = 0.4) -> np.ndarray:
    original = np.array(original_image.resize(IMG_SIZE))
    heatmap_resized = cv2.resize(heatmap, IMG_SIZE)
    colored = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(original, 1 - alpha, colored, alpha, 0)