import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

MODEL_DIR = Path(os.environ.get("MODEL_DIR", BASE_DIR / "models"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

HUGGINGFACE_REPO_ID = os.environ.get("HUGGINGFACE_REPO_ID", "")
HUGGINGFACE_MODEL_FILENAME = os.environ.get(
    "HUGGINGFACE_MODEL_FILENAME", "brain_tumor_model.pth"
)
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN") or None

MODEL_PATH = MODEL_DIR / HUGGINGFACE_MODEL_FILENAME
LABELS_PATH = Path(os.environ.get("LABELS_PATH", BASE_DIR / "labels.txt"))
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", BASE_DIR / "uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*")

TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
