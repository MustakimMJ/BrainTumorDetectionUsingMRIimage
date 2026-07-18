import logging

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from torchvision import models, transforms

from app.config import (
    HUGGINGFACE_MODEL_FILENAME,
    HUGGINGFACE_REPO_ID,
    HUGGINGFACE_TOKEN,
    LABELS_PATH,
    MODEL_DIR,
    MODEL_PATH,
)

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None
class_names = []
preprocess = None


def _download_model_if_needed() -> None:
    if MODEL_PATH.is_file():
        logger.info("Model already exists at %s; skipping download.", MODEL_PATH)
        return

    if not HUGGINGFACE_REPO_ID:
        raise RuntimeError(
            "Model file not found and HUGGINGFACE_REPO_ID is not set. "
            "Provide a Hugging Face repository ID or place the model locally."
        )

    logger.info(
        "Downloading model '%s' from Hugging Face repo '%s'.",
        HUGGINGFACE_MODEL_FILENAME,
        HUGGINGFACE_REPO_ID,
    )
    hf_hub_download(
        repo_id=HUGGINGFACE_REPO_ID,
        filename=HUGGINGFACE_MODEL_FILENAME,
        local_dir=str(MODEL_DIR),
        token=HUGGINGFACE_TOKEN,
    )
    logger.info("Model downloaded to %s.", MODEL_PATH)


def _load_class_names() -> list[str]:
    if not LABELS_PATH.is_file():
        raise FileNotFoundError(f"Labels file not found: {LABELS_PATH}")

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def initialize_model() -> None:
    global model, class_names, preprocess

    _download_model_if_needed()
    class_names = _load_class_names()

    loaded_model = models.resnet18(pretrained=False)
    num_features = loaded_model.fc.in_features
    loaded_model.fc = nn.Linear(num_features, len(class_names))

    try:
        state = torch.load(MODEL_PATH, map_location=device)
        loaded_model.load_state_dict(state, strict=False)
    except Exception as exc:
        logger.exception("Error loading model weights from %s", MODEL_PATH)
        raise RuntimeError(f"Failed to load model weights: {exc}") from exc

    loaded_model.to(device)
    loaded_model.eval()
    model = loaded_model

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    logger.info(
        "Model loaded on %s with %d classes.",
        device,
        len(class_names),
    )


def is_model_ready() -> bool:
    return model is not None and preprocess is not None and bool(class_names)
