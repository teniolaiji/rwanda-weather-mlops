"""Deploy the API + dashboard to Hugging Face Spaces (free Docker hosting, 2 vCPU / 16 GB RAM).

One-time setup:
    pip install huggingface_hub
    hf auth login          (paste a token with "write" access from huggingface.co/settings/tokens)
Deploy or update:
    python deploy/hf_deploy.py
The Space builds the Dockerfile and serves the app at https://<user>-rwanda-weather-mlops.hf.space
"""
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
NAME = sys.argv[1] if len(sys.argv) > 1 else "rwanda-weather-mlops"

SPACE_README = """---
title: Rwanda Weather MLOps
emoji: 🌦️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 8000
pinned: false
short_description: Weather photo classifier with upload, retraining and monitoring
---

Classifies weather photos (fog/smog, hail, lightning, rain, rainbow) with a fine-tuned MobileNetV2.
Upload new labelled images and retrain from the dashboard. API docs at `/docs`.
"""

api = HfApi()
user = api.whoami()["name"]
repo_id = f"{user}/{NAME}"

api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
api.upload_file(path_or_fileobj=SPACE_README.encode("utf-8"), path_in_repo="README.md",
                repo_id=repo_id, repo_type="space", commit_message="Space config")
api.upload_folder(
    folder_path=ROOT, repo_id=repo_id, repo_type="space", commit_message="Deploy app",
    allow_patterns=["Dockerfile", ".dockerignore", "requirements-api.txt",
                    "src/*.py", "api/*.py", "api/static/*", "api/static/fonts/*",
                    "models/*.h5", "models/*.json", "data/train/**", "data/test/**"],
    ignore_patterns=["**/__pycache__/**"],
)

sub = f"{user}-{NAME}".lower().replace("_", "-").replace(".", "-")
print(f"\nUploaded. The Space is building now (about 5 to 10 minutes).")
print(f"  Build logs: https://huggingface.co/spaces/{repo_id}")
print(f"  Live app:   https://{sub}.hf.space")
print(f"  API docs:   https://{sub}.hf.space/docs")
