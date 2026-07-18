import io
import logging
import os

from flask import Blueprint, jsonify, render_template, request, send_from_directory
import torch
from PIL import Image
from werkzeug.utils import secure_filename

from app import model as model_module
from app.config import UPLOAD_DIR

logger = logging.getLogger(__name__)

bp = Blueprint("main", __name__)


@bp.route("/")
def index():
    return render_template("index.html")


@bp.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})


@bp.route("/predict", methods=["POST"])
def predict():
    if not model_module.is_model_ready():
        logger.error("Prediction requested before model was initialized.")
        return jsonify({"error": "Model is not ready"}), 503

    if "image" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    img_bytes = file.read()
    try:
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as exc:
        logger.warning("Invalid image upload: %s", exc)
        return jsonify({"error": "Invalid image file", "detail": str(exc)}), 400

    safe_filename = secure_filename(file.filename) or "upload.jpg"
    save_path = os.path.join(UPLOAD_DIR, safe_filename)
    try:
        with open(save_path, "wb") as f:
            f.write(img_bytes)
    except OSError as exc:
        logger.warning("Could not save uploaded image to %s: %s", save_path, exc)

    input_tensor = model_module.preprocess(image).unsqueeze(0).to(model_module.device)

    with torch.no_grad():
        logits = model_module.model(input_tensor)
        probs = torch.softmax(logits, dim=1).cpu().squeeze(0).tolist()
        best_idx = int(torch.argmax(logits, dim=1).item())
        best_prob = probs[best_idx]

    response = {
        "prediction": model_module.class_names[best_idx],
        "confidence": round(best_prob * 100, 2),
        "all_probs": {
            model_module.class_names[i]: round(p * 100, 2)
            for i, p in enumerate(probs)
        },
        "filename": safe_filename,
    }
    logger.info(
        "Prediction complete: %s (%.2f%%)",
        response["prediction"],
        response["confidence"],
    )
    return jsonify(response)


@bp.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)
