"""Model building, evaluation and retraining.

Retraining always starts from the CURRENT saved model (models/weather_model.h5), never from
scratch, so the model keeps what it already knows and fine-tunes on old + new data.
A retrained model only replaces the current one if its macro F1 on the test set is at
least as good (within a small tolerance). Otherwise the current model stays in production.

CLI (local test):  python -m src.model --retrain data/retrain_demo
"""
import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.preprocessing import DATA_DIR, IMG_SIZE, MODELS_DIR, SEED, ALLOWED_EXTENSIONS, load_class_names

MODEL_PATH = MODELS_DIR / "weather_model.h5"
METRICS_PATH = MODELS_DIR / "metrics.json"
VERSIONS_DIR = MODELS_DIR / "versions"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"


# ---------- data ----------

def list_images(directory, class_names):
    """Folder of class sub-folders -> list of (path, label_index). Missing class folders are fine."""
    items = []
    for idx, c in enumerate(class_names):
        folder = Path(directory) / c
        if folder.is_dir():
            items += [(str(p), idx) for p in sorted(folder.iterdir()) if p.suffix.lower() in ALLOWED_EXTENSIONS]
    return items


def make_dataset(items, batch_size=32, shuffle=False, augment=False):
    """List of (path, label_index) -> tf.data.Dataset of (raw 0-255 images, labels)."""
    import tensorflow as tf
    from src.preprocessing import build_augmenter

    paths = [p for p, _ in items]
    labels = [l for _, l in items]

    def load(path, label):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        return tf.cast(img, tf.float32), label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(items), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)
    if augment:
        aug = build_augmenter()
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


# ---------- model ----------

def build_model(num_classes, dropout=0.3, l2=1e-4, weights="imagenet"):
    """Same architecture as the notebook: MobileNetV2 base + regularised head."""
    import tensorflow as tf
    from tensorflow.keras import layers, regularizers

    base = tf.keras.applications.MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights=weights)
    base.trainable = False
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")
    x = layers.Rescaling(1.0 / 127.5, offset=-1, name="normalise")(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return tf.keras.Model(inputs, outputs, name="weather_mobilenetv2")


def load_model(path=MODEL_PATH):
    import tensorflow as tf
    return tf.keras.models.load_model(path, compile=False)


def evaluate(model, items, class_names):
    """Same 7 metrics as the notebook, plus per-class scores and the confusion matrix."""
    import tensorflow as tf
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                                 roc_auc_score, classification_report, confusion_matrix)

    y_true = np.array([l for _, l in items])
    probs = model.predict(make_dataset(items), verbose=0)
    y_pred = probs.argmax(axis=1)
    labels = list(range(len(class_names)))
    metrics = {
        "loss": float(tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_true, probs))),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "roc_auc_ovr": roc_auc_score(y_true, probs, multi_class="ovr", labels=labels),
    }
    report = classification_report(y_true, y_pred, labels=labels, target_names=class_names,
                                   output_dict=True, zero_division=0)
    return {
        "test_metrics": {k: round(float(v), 4) for k, v in metrics.items()},
        "per_class": {c: {k: round(float(report[c][k]), 4) for k in ["precision", "recall", "f1-score", "support"]}
                      for c in class_names},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def read_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            return json.load(f)
    return {}


# ---------- retraining ----------

def retrain(new_items, epochs=3, learning_rate=1e-5, tolerance=0.01, replay_per_class=None, log=print):
    """Fine-tune the current model on existing training data + new labelled images.

    new_items: list of (image_path, class_name) for the newly uploaded images.
    replay_per_class: if set, mix the new images with only this many old images per class
        (a "replay buffer"). This keeps retraining fast on a small CPU server while still
        reminding the model of what it already knows, so it does not forget old classes.
    Returns a report dict. Saves the new model only if it is not worse than the current one.
    """
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight

    class_names = load_class_names()
    started = datetime.now(timezone.utc)

    # 1. Data: existing train images + new uploads, with a stratified validation split for early stopping
    new = [(str(p), class_names.index(c)) for p, c in new_items if c in class_names]
    if not new:
        raise ValueError("No new labelled images to retrain on.")
    old_all = list_images(TRAIN_DIR, class_names)
    old = old_all
    if replay_per_class:
        rng = np.random.default_rng(SEED)
        old = []
        for idx in range(len(class_names)):
            cls = [it for it in old_all if it[1] == idx]
            take = rng.choice(len(cls), size=min(replay_per_class, len(cls)), replace=False) if cls else []
            old += [cls[i] for i in take]
    test = list_images(TEST_DIR, class_names)
    all_items = old + new
    labels = [l for _, l in all_items]
    train_items, val_items = train_test_split(all_items, test_size=0.15, stratify=labels, random_state=SEED)
    log(f"Retraining on {len(train_items)} images ({len(new)} new), validating on {len(val_items)}")

    present = np.unique([l for _, l in train_items])
    weights = compute_class_weight("balanced", classes=present, y=[l for _, l in train_items])
    class_weight = {int(k): float(v) for k, v in zip(present, weights)}

    # 2. Baseline: the current production model's score on the untouched test set
    current = read_metrics()
    baseline = current.get("test_metrics") or evaluate(load_model(), test, class_names)["test_metrics"]

    # 3. Load the current model as the pretrained starting point and fine-tune it
    log(f"Baseline macro F1 of current model: {baseline['f1_macro']:.4f}")
    log(f"Loading current model ({MODEL_PATH.name}) as the pretrained starting point")
    model = load_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(
        make_dataset(train_items, shuffle=True, augment=True),
        validation_data=make_dataset(val_items),
        epochs=epochs,
        class_weight=class_weight,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
            tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda e, m: log(
                f"Epoch {e + 1}/{epochs}: loss {m['loss']:.4f}, accuracy {m['accuracy']:.4f}, "
                f"val_loss {m['val_loss']:.4f}, val_accuracy {m['val_accuracy']:.4f}")),
        ],
        verbose=2,
    )

    # 4. Evaluate on the same test set and decide
    log(f"Evaluating retrained model on {len(test)} untouched test images")
    result = evaluate(model, test, class_names)
    new_f1, old_f1 = result["test_metrics"]["f1_macro"], baseline["f1_macro"]
    accepted = new_f1 >= old_f1 - tolerance
    version = int(current.get("model_version", 1))

    if accepted:
        # keep a copy of the old model, then promote the new one
        VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
        if MODEL_PATH.exists():
            shutil.copy2(MODEL_PATH, VERSIONS_DIR / f"weather_model_v{version}.h5")
        version += 1
        model.save(MODEL_PATH)
        # new images become part of the permanent training set
        for path, idx in new:
            dst = TRAIN_DIR / class_names[idx] / Path(path).name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(path, dst)

    report = {
        "accepted": accepted,
        "model_version": version,
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "new_images": len(new),
        "epochs_run": len(history.history["loss"]),
        "previous_metrics": baseline,
        "new_metrics": result["test_metrics"],
        "reason": (f"Accepted: macro F1 {new_f1:.4f} vs previous {old_f1:.4f}" if accepted
                   else f"Rejected: macro F1 {new_f1:.4f} is worse than previous {old_f1:.4f}, kept current model"),
    }

    if accepted:
        retrain_history = current.get("retrain_history", []) + [{k: report[k] for k in
                          ["model_version", "finished_at", "new_images", "previous_metrics", "new_metrics"]}]
        current.update({
            "model_version": version,
            "trained_at": report["finished_at"],
            "tensorflow_version": tf.__version__,
            "n_train_images": len(old_all) + len(new),
            "retrain_history": retrain_history,
            **result,
        })
        with open(METRICS_PATH, "w") as f:
            json.dump(current, f, indent=2)

    log(report["reason"])
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain the weather model on a folder of new labelled images.")
    parser.add_argument("--retrain", required=True, help="Folder with one sub-folder per class, e.g. data/retrain_demo")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    names = load_class_names()
    items = [(p, names[l]) for p, l in list_images(args.retrain, names)]
    print(json.dumps(retrain(items, epochs=args.epochs), indent=2))
