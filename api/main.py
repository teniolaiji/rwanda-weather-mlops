"""FastAPI app: prediction, bulk upload, retraining, monitoring and the web dashboard.

Run from the repo root:  uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
import io
import json
import os
import socket
import threading
import time
import traceback
import uuid
import zipfile
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api import db
from src.model import METRICS_PATH, retrain as run_retrain
from src.prediction import get_predictor
from src.preprocessing import ALLOWED_EXTENSIONS, DATA_DIR, MODELS_DIR, load_class_names, resize_and_save

RETRAIN_THRESHOLD = int(os.getenv("RETRAIN_THRESHOLD", "50"))     # auto-retrain once this many new images are waiting
RETRAIN_EPOCHS = int(os.getenv("RETRAIN_EPOCHS", "3"))
REPLAY_PER_CLASS = int(os.getenv("REPLAY_PER_CLASS", "120")) or None
UPLOAD_DIR = DATA_DIR / "uploads"
STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_FILE_MB = 10

START_TIME = time.time()
STARTED_AT = datetime.now(timezone.utc).isoformat(timespec="seconds")
CONTAINER_ID = socket.gethostname()           # shows which Docker container answered (useful under load balancing)

job = {"state": "idle", "run_id": None, "trigger": None, "logs": [], "report": None, "started_at": None}
job_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app):
    db.init_db()
    db.fail_stale_runs()
    get_predictor()                            # load the model once, before the first request
    yield


app = FastAPI(title="Rwanda Weather MLOps API",
              description="Weather image classification: predict, upload new data, retrain and monitor.",
              version="1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
# test images only (never the whole data folder, which holds the database)
app.mount("/test-images", StaticFiles(directory=DATA_DIR / "test"), name="test-images")


# ---------- pages ----------

@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(STATIC_DIR / "index.html")


# ---------- monitoring ----------

@app.get("/health", tags=["Monitoring"])
def health():
    up = int(time.time() - START_TIME)
    metrics = _read_json(METRICS_PATH)
    return {
        "status": "ok",
        "container": CONTAINER_ID,
        "started_at": STARTED_AT,
        "uptime_seconds": up,
        "uptime": f"{up // 86400}d {up % 86400 // 3600}h {up % 3600 // 60}m {up % 60}s",
        "model_version": metrics.get("model_version", 1),
        "model_trained_at": metrics.get("trained_at"),
        "retraining": job["state"] == "running",
    }


@app.get("/metrics", tags=["Monitoring"])
def metrics():
    return _read_json(METRICS_PATH)


@app.get("/stats", tags=["Monitoring"])
def stats():
    return {"predictions": db.prediction_stats(), "uploads": db.upload_totals(), "retrain_runs": db.list_runs(10)}


@app.get("/insights", tags=["Data"])
def insights():
    data = _read_json(MODELS_DIR / "insights.json")
    names = load_class_names()
    data["current_train_counts"] = {c: len(list((DATA_DIR / "train" / c).glob("*.jpg"))) for c in names}
    return data


@app.get("/samples", tags=["Data"])
def samples(per_class: int = 2):
    """Random test-set photos (never seen in training) with their true labels, for trying the model."""
    import random
    out = []
    for c in load_class_names():
        files = sorted((DATA_DIR / "test" / c).glob("*.jpg"))
        for f in random.sample(files, min(per_class, len(files))):
            out.append({"label": c, "url": f"/test-images/{c}/{f.name}"})
    return out


# ---------- prediction ----------

@app.post("/predict", tags=["Prediction"])
def predict(file: UploadFile = File(..., description="A weather photo (jpg, png, ...)")):
    data = file.file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    try:
        result = get_predictor().predict_bytes(data)
    except Exception:
        raise HTTPException(400, "Could not read this file as an image")
    db.log_prediction(result["label"], result["confidence"], result["latency_ms"])
    return {**result, "container": CONTAINER_ID}


# ---------- upload ----------

@app.post("/upload", tags=["Retraining"])
def upload(files: List[UploadFile] = File(..., description="Images, or a .zip with one folder per class"),
           label: Optional[str] = Form(None, description="Class for loose images (ignored for zips with class folders)")):
    names = load_class_names()
    label = label.strip().lower() if label else None
    if label and label not in names:
        raise HTTPException(400, f"label must be one of {names}")

    saved, rejected = [], []
    for f in files:
        data = f.file.read()
        if f.filename.lower().endswith(".zip"):
            for name, content, zlabel in _iter_zip(data, names):
                _save_one(name, content, zlabel or label, saved, rejected)
        else:
            _save_one(f.filename, data, label, saved, rejected)

    auto = _maybe_auto_retrain()
    return {"saved": len(saved), "rejected": rejected, "saved_by_class": _count(saved),
            "pending_by_class": db.pending_counts(), "pending_total": db.upload_totals()["pending"],
            "auto_retrain_threshold": RETRAIN_THRESHOLD, "auto_retrain_started": auto}


@app.get("/uploads", tags=["Retraining"])
def uploads():
    return {"totals": db.upload_totals(), "pending_by_class": db.pending_counts(),
            "auto_retrain_threshold": RETRAIN_THRESHOLD, "recent": db.recent_uploads()}


def _iter_zip(data, names):
    """Yield (filename, bytes, label-from-folder-or-None) for each image inside a zip."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        return
    for info in zf.infolist():
        p = Path(info.filename)
        if info.is_dir() or "__MACOSX" in p.parts or p.name.startswith("."):
            continue
        folder = p.parent.name.lower()
        yield p.name, zf.read(info), (folder if folder in names else None)


def _save_one(name, data, label, saved, rejected):
    if Path(name).suffix.lower() not in ALLOWED_EXTENSIONS:
        rejected.append({"file": name, "reason": "not an image file"}); return
    if not label:
        rejected.append({"file": name, "reason": "no label: choose a class or use class folders in the zip"}); return
    if len(data) > MAX_FILE_MB * 1e6:
        rejected.append({"file": name, "reason": f"larger than {MAX_FILE_MB} MB"}); return
    dst = UPLOAD_DIR / label / f"upload_{uuid.uuid4().hex[:12]}.jpg"
    try:
        resize_and_save(data, dst)              # preprocessing: EXIF fix, RGB, 224x224 JPEG
    except Exception:
        rejected.append({"file": name, "reason": "corrupt or unreadable image"}); return
    db.add_upload(name, label, dst.read_bytes(), dst)
    saved.append(label)


def _count(labels):
    out = {}
    for l in labels:
        out[l] = out.get(l, 0) + 1
    return out


# ---------- retraining ----------

@app.post("/retrain", tags=["Retraining"])
def retrain():
    if not _start_retrain("manual"):
        if job["state"] == "running":
            raise HTTPException(409, "Retraining is already running")
        raise HTTPException(400, "No new uploaded images to retrain on. Upload data first.")
    return {"message": "Retraining started", "run_id": job["run_id"]}


@app.get("/retrain/status", tags=["Retraining"])
def retrain_status():
    return {**{k: job[k] for k in ["state", "run_id", "trigger", "started_at", "report"]},
            "logs": job["logs"][-50:], "history": db.list_runs(10)}


def _maybe_auto_retrain():
    return db.upload_totals()["pending"] >= RETRAIN_THRESHOLD and _start_retrain("auto")


def _start_retrain(trigger):
    with job_lock:
        if job["state"] == "running" or db.upload_totals()["pending"] == 0:
            return False
        run_id = db.start_run(trigger)
        job.update(state="running", run_id=run_id, trigger=trigger, logs=[], report=None,
                   started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    threading.Thread(target=_retrain_worker, args=(run_id,), daemon=True).start()
    return True


def _log(msg):
    job["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def _retrain_worker(run_id):
    try:
        rows = db.pending_uploads()
        _log(f"Retraining triggered ({job['trigger']}): {len(rows)} new images found in the database")

        # Preprocessing of uploaded data: restore any missing files from the database copy
        _log("Preprocessing uploaded images (224x224 RGB, already cleaned at upload; augmentation applied in training)")
        for r in rows:
            p = Path(r["file_path"])
            if not p.exists():
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(r["image"])
        items = [(r["file_path"], r["label"]) for r in rows]

        report = run_retrain(items, epochs=RETRAIN_EPOCHS, replay_per_class=REPLAY_PER_CLASS, log=_log)
        db.mark_used([r["id"] for r in rows], run_id)
        if report["accepted"]:
            get_predictor().reload()
            _log(f"New model v{report['model_version']} is now live")
        db.finish_run(run_id, "completed", report)
        job.update(state="completed", report=report)
    except Exception as e:
        traceback.print_exc()
        _log(f"Retraining failed: {e}")
        db.finish_run(run_id, "failed", message=str(e))
        job.update(state="failed")


def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
