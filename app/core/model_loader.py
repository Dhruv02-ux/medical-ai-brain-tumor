"""Loads the Keras model once (singleton) — avoids reloading on every request."""
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
try:
    import tf_keras as keras
except ImportError:
    import tensorflow.keras as keras

from app.core.config import MODEL_PATH

def _patch_layer_init() -> None:
    """Compat fix: saved model passes 'quantization_config' kwarg this
    Keras build's Layer.__init__ doesn't accept — strip it before load."""
    for module in (keras, tf.keras):
        try:
            original_init = module.layers.Layer.__init__
            def patched_init(self, *args, _orig=original_init, **kwargs):
                kwargs.pop("quantization_config", None)
                _orig(self, *args, **kwargs)
            module.layers.Layer.__init__ = patched_init
        except Exception:
            pass

_patch_layer_init()
_model = None
_load_error: str | None = None

def get_model():
    """Returns the cached model, loading it on first call only."""
    global _model, _load_error
    if _model is None and _load_error is None:
        try:
            _model = keras.models.load_model(MODEL_PATH, compile=False)
        except Exception as e:
            _load_error = str(e)
    if _model is None:
        raise RuntimeError(f"Model failed to load: {_load_error}")
    return _model