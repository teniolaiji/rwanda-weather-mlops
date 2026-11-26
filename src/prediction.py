"""Single-image prediction.

The model is loaded once and kept in memory, so each request only pays for inference.
After retraining, reload() swaps in the new model without restarting the server.
"""
import threading
import time

import numpy as np

from src.preprocessing import load_class_names, preprocess_bytes, preprocess_path
from src.model import MODEL_PATH, load_model


class Predictor:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self._lock = threading.Lock()
        self.reload()

    def reload(self):
        model = load_model(self.model_path)
        names = load_class_names()
        with self._lock:                       # swap atomically so in-flight requests never see a half-loaded model
            self.model, self.class_names = model, names

    def _predict_array(self, arr):
        start = time.perf_counter()
        with self._lock:
            model, names = self.model, self.class_names
        probs = model(arr, training=False).numpy()[0]
        idx = int(np.argmax(probs))
        return {
            "label": names[idx],
            "confidence": round(float(probs[idx]), 4),
            "probabilities": {c: round(float(p), 4) for c, p in zip(names, probs)},
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
        }

    def predict_bytes(self, data):
        return self._predict_array(preprocess_bytes(data))

    def predict_path(self, path):
        return self._predict_array(preprocess_path(path))


_predictor = None


def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = Predictor()
    return _predictor


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(get_predictor().predict_path(sys.argv[1]), indent=2))
