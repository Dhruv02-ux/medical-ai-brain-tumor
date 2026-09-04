"""Loads the Keras model once (singleton) — avoids reloading on every request."""
import os
import threading
import numpy as np
import tensorflow as tf

from app.core.config import MODEL_PATH

_model = None
_load_error: str | None = None
_lock = threading.Lock()

def get_model() -> tf.keras.Model:
    """Returns the cached model, loading and warming it up on first call only."""
    global _model, _load_error
    if _model is None:
        with _lock:
            if _model is None and _load_error is None:
                try:
                    if not os.path.exists(MODEL_PATH):
                        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
                    _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                    # Warm up computation graph for sub-50ms user response
                    dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
                    _model(dummy)
                except Exception as e:
                    _load_error = str(e)
    if _model is None:
        raise RuntimeError(f"Model failed to load: {_load_error}")
    return _model