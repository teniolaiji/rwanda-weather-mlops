"""SQLite database for uploaded training images, retraining runs and prediction logs.

Uploaded images are stored in the database itself (preprocessed JPEG bytes as a BLOB) together
with their label and whether they have been used for retraining yet. A copy is also written to
data/uploads/ so TensorFlow can read it quickly during retraining.
"""
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.getenv("DB_PATH", ROOT / "data" / "app.db"))
_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS uploads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    original_name TEXT,
    label TEXT NOT NULL,
    image BLOB NOT NULL,
    file_path TEXT NOT NULL,
    uploaded_at TEXT NOT NULL,
    used_in_training INTEGER NOT NULL DEFAULT 0,
    retrain_run_id INTEGER
);
CREATE TABLE IF NOT EXISTS retrain_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trigger TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    new_images INTEGER,
    accepted INTEGER,
    model_version INTEGER,
    previous_f1 REAL,
    new_f1 REAL,
    previous_accuracy REAL,
    new_accuracy REAL,
    message TEXT
);
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    label TEXT NOT NULL,
    confidence REAL NOT NULL,
    latency_ms REAL NOT NULL
);
"""


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect():
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with connect() as c:
        c.execute("PRAGMA journal_mode=WAL")       # lets reads continue while a write happens
        c.executescript(SCHEMA)


# ---------- uploads ----------

def add_upload(original_name, label, image_bytes, file_path):
    with _lock, connect() as c:
        cur = c.execute(
            "INSERT INTO uploads (original_name, label, image, file_path, uploaded_at) VALUES (?, ?, ?, ?, ?)",
            (original_name, label, image_bytes, str(file_path), now()))
        return cur.lastrowid


def pending_uploads():
    with connect() as c:
        return [dict(r) for r in c.execute(
            "SELECT id, label, file_path, image FROM uploads WHERE used_in_training = 0 ORDER BY id")]


def pending_counts():
    with connect() as c:
        rows = c.execute("SELECT label, COUNT(*) n FROM uploads WHERE used_in_training = 0 GROUP BY label")
        return {r["label"]: r["n"] for r in rows}


def upload_totals():
    with connect() as c:
        r = c.execute("SELECT COUNT(*) total, COALESCE(SUM(used_in_training), 0) used FROM uploads").fetchone()
        return {"total": r["total"], "used": r["used"], "pending": r["total"] - r["used"]}


def recent_uploads(limit=20):
    with connect() as c:
        return [dict(r) for r in c.execute(
            "SELECT id, original_name, label, uploaded_at, used_in_training, retrain_run_id "
            "FROM uploads ORDER BY id DESC LIMIT ?", (limit,))]


def mark_used(upload_ids, run_id):
    with _lock, connect() as c:
        c.executemany("UPDATE uploads SET used_in_training = 1, retrain_run_id = ? WHERE id = ?",
                      [(run_id, i) for i in upload_ids])


# ---------- retraining runs ----------

def start_run(trigger):
    with _lock, connect() as c:
        return c.execute("INSERT INTO retrain_runs (trigger, status, started_at) VALUES (?, 'running', ?)",
                         (trigger, now())).lastrowid


def finish_run(run_id, status, report=None, message=None):
    report = report or {}
    prev, new = report.get("previous_metrics", {}), report.get("new_metrics", {})
    with _lock, connect() as c:
        c.execute(
            "UPDATE retrain_runs SET status=?, finished_at=?, new_images=?, accepted=?, model_version=?, "
            "previous_f1=?, new_f1=?, previous_accuracy=?, new_accuracy=?, message=? WHERE id=?",
            (status, now(), report.get("new_images"), int(report["accepted"]) if "accepted" in report else None,
             report.get("model_version"), prev.get("f1_macro"), new.get("f1_macro"),
             prev.get("accuracy"), new.get("accuracy"), message or report.get("reason"), run_id))


def list_runs(limit=20):
    with connect() as c:
        return [dict(r) for r in c.execute("SELECT * FROM retrain_runs ORDER BY id DESC LIMIT ?", (limit,))]


def fail_stale_runs():
    """A run still marked 'running' at startup was interrupted by a restart."""
    with _lock, connect() as c:
        c.execute("UPDATE retrain_runs SET status='failed', finished_at=?, message='Interrupted by server restart' "
                  "WHERE status='running'", (now(),))


# ---------- predictions ----------

def log_prediction(label, confidence, latency_ms):
    try:
        with connect() as c:
            c.execute("INSERT INTO predictions (created_at, label, confidence, latency_ms) VALUES (?, ?, ?, ?)",
                      (now(), label, confidence, latency_ms))
    except sqlite3.OperationalError:
        pass   # never fail a prediction because the log is busy under heavy load


def prediction_stats():
    with connect() as c:
        r = c.execute("SELECT COUNT(*) n, AVG(latency_ms) avg_ms, AVG(confidence) avg_conf FROM predictions").fetchone()
        by_label = {x["label"]: x["n"] for x in c.execute("SELECT label, COUNT(*) n FROM predictions GROUP BY label")}
        return {"total": r["n"], "avg_latency_ms": round(r["avg_ms"] or 0, 1),
                "avg_confidence": round(r["avg_conf"] or 0, 4), "by_label": by_label}
