"""Flood the API with prediction requests.

Each simulated user keeps sending real test images to /predict (and occasionally checks /health),
waiting 0.5 to 1.5 s between requests, like people using the app at the same time.

Interactive (web UI on http://localhost:8089):
    python -m locust -f locust/locustfile.py --host http://localhost:8080
Headless (used by run_experiments.ps1):
    python -m locust -f locust/locustfile.py --host http://localhost:8080 --headless -u 30 -r 10 -t 60s --csv results/c1
"""
import random
from collections import Counter
from pathlib import Path

from locust import HttpUser, between, events, task

IMAGES = sorted((Path(__file__).resolve().parent.parent / "data" / "test").glob("*/*.jpg"))
if not IMAGES:
    raise SystemExit("No images in data/test. Run this from the repo after the notebook has produced the data.")

served_by = Counter()      # which container answered: proves the load balancer spreads the work


class WeatherUser(HttpUser):
    wait_time = between(0.5, 1.5)

    def on_start(self):
        self.images = [p.read_bytes() for p in random.sample(IMAGES, min(25, len(IMAGES)))]

    @task(10)
    def predict(self):
        with self.client.post("/predict", files={"file": ("photo.jpg", random.choice(self.images), "image/jpeg")},
                              name="/predict", catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}")
                return
            served_by[r.json().get("container", "unknown")] += 1
            r.success()

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")


@events.test_stop.add_listener
def report_containers(environment, **kwargs):
    if served_by:
        total = sum(served_by.values())
        print("\nPredictions answered per container:")
        for name, n in served_by.most_common():
            print(f"  {name}: {n} ({n / total:.0%})")
