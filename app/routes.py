import io
import logging
import os
import time
import traceback
import numpy as np
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
    overall_start = time.time()

    try:
        logger.info("=" * 60)
        logger.info("Prediction request received")

        # -----------------------------
        # Check model
        # -----------------------------
        if not model_module.is_model_ready():
            logger.error("Model is not initialized.")
            return jsonify({"error": "Model is not ready"}), 503

        logger.info("Model is ready.")

        # -----------------------------
        # Validate uploaded file
        # -----------------------------
        if "image" not in request.files:
            logger.warning("No image found in request.")
            return jsonify({"error": "No file part"}), 400

        file = request.files["image"]

        if file.filename == "":
            logger.warning("Empty filename.")
            return jsonify({"error": "No selected file"}), 400

        logger.info(f"Uploaded file: {file.filename}")

        # -----------------------------
        # Read image
        # -----------------------------
        img_bytes = file.read()

        logger.info("Image size: %.2f KB", len(img_bytes) / 1024)

        try:
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            logger.info("Image opened successfully.")
        except Exception as exc:
            logger.exception("Failed to open image.")
            return jsonify({
                "error": "Invalid image",
                "detail": str(exc)
            }), 400

        # -----------------------------
        # Save uploaded image
        # -----------------------------
        safe_filename = secure_filename(file.filename) or "upload.jpg"

        save_path = os.path.join(UPLOAD_DIR, safe_filename)

        try:
            with open(save_path, "wb") as f:
                f.write(img_bytes)

            logger.info("Saved uploaded image to %s", save_path)

        except Exception as exc:
            logger.warning("Could not save uploaded image: %s", exc)

        # -----------------------------
        # Preprocessing
        # -----------------------------
        logger.info("Starting manual preprocessing...")

        t0 = time.time()

        image = image.resize((224, 224))

        img = np.array(image).astype(np.float32)
        img = img / 255.0

        img[:, :, 0] = (img[:, :, 0] - 0.485) / 0.229
        img[:, :, 1] = (img[:, :, 1] - 0.456) / 0.224
        img[:, :, 2] = (img[:, :, 2] - 0.406) / 0.225

        img = img.transpose((2, 0, 1))

        input_tensor = torch.from_numpy(img).float()
        input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.to(model_module.device)

        logger.info("Manual preprocessing completed in %.3f sec", time.time() - t0)

        logger.info("Input tensor shape: %s", tuple(input_tensor.shape))
        logger.info("Device: %s", model_module.device)

        # -----------------------------
        # Inference
        # -----------------------------
        logger.info("Starting inference...")

        t1 = time.time()

        with torch.no_grad():
            logits = model_module.model(input_tensor)

        logger.info(
            "Inference completed in %.3f sec",
            time.time() - t1
        )

        logger.info("Logits shape: %s", tuple(logits.shape))

        # -----------------------------
        # Softmax
        # -----------------------------
        probs = torch.softmax(logits, dim=1).cpu().squeeze(0).tolist()

        best_idx = int(torch.argmax(logits, dim=1).item())
        best_prob = float(probs[best_idx])

        logger.info("Predicted class index: %d", best_idx)

        if best_idx >= len(model_module.class_names):
            logger.error(
                "Predicted index %d exceeds number of labels (%d)",
                best_idx,
                len(model_module.class_names),
            )

            return jsonify({
                "error": "Label index out of range"
            }), 500

        # -----------------------------
        # Response
        # -----------------------------
        response = {
            "prediction": model_module.class_names[best_idx],
            "confidence": round(best_prob * 100, 2),
            "all_probs": {
                model_module.class_names[i]: round(float(p) * 100, 2)
                for i, p in enumerate(probs)
            },
            "filename": safe_filename,
        }

        logger.info("Prediction: %s", response["prediction"])
        logger.info("Confidence: %.2f%%", response["confidence"])

        logger.info(
            "Total request time: %.3f sec",
            time.time() - overall_start,
        )

        logger.info("=" * 60)

        return jsonify(response)

    except Exception as exc:
        logger.error("=" * 60)
        logger.exception("Prediction failed")
        logger.error(traceback.format_exc())
        logger.error("=" * 60)

        return jsonify({
            "status": "error",
            "message": str(exc),
            "type": type(exc).__name__,
        }), 500


@bp.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)