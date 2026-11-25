"""Preprocessing shared by prediction, uploads and retraining.

The same steps as the notebook: EXIF-rotate, convert to RGB, resize to 224x224.
Normalisation to [-1, 1] happens inside the model (Rescaling layer), so these
functions return raw 0-255 pixels.
"""
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
IMG_SIZE = 224
SEED = 42
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_class_names():
    with open(MODELS_DIR / "class_names.json") as f:
        return json.load(f)


def _clean(img):
    """Fix phone-camera rotation, drop alpha/greyscale, resize."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    return img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)


def preprocess_bytes(data):
    """Raw uploaded bytes -> array of shape (1, 224, 224, 3) ready for model.predict."""
    with Image.open(io.BytesIO(data)) as img:
        arr = np.asarray(_clean(img), dtype=np.float32)
    return arr[None, ...]


def preprocess_path(path):
    with open(path, "rb") as f:
        return preprocess_bytes(f.read())


def resize_and_save(data, dst):
    """Clean an uploaded image and save it as a 224x224 JPEG (used for retraining uploads).

    Raises an exception if the bytes are not a readable image, so bad files are rejected.
    """
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(io.BytesIO(data)) as img:
        _clean(img).save(dst, "JPEG", quality=90)
    return dst


def build_augmenter():
    """Training-only augmentation. No vertical flips (sky is up) and no colour shifts (colour is a key feature)."""
    import tensorflow as tf
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomContrast(0.2),
    ], name="augment")


def make_dataset(directory, class_names, batch_size=32, shuffle=True, augment=False):
    """Folder of class sub-folders -> tf.data.Dataset of (images, int labels)."""
    import tensorflow as tf
    ds = tf.keras.utils.image_dataset_from_directory(
        str(directory), image_size=(IMG_SIZE, IMG_SIZE), batch_size=batch_size,
        label_mode="int", class_names=class_names, shuffle=shuffle, seed=SEED,
    )
    if augment:
        aug = build_augmenter()
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)
