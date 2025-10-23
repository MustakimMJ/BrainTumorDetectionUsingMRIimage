import os
import io
from flask import Flask, request, jsonify, render_template, send_from_directory
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# Configuration
MODEL_PATH = "brain_tumor_model.pth"  # adjust if different filename
LABELS_PATH = "labels.txt"
IMAGE_SAVE_DIR = "uploads"
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

# Load labels
with open(LABELS_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Build ResNet18 model and load weights
model = models.resnet18(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_names))  # Replace the last fc layer

try:
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state, strict=False)  # allow some mismatch tolerance
except Exception as e:
    print(f"Error loading model weights: {e}")

model.to(device)
model.eval()

# Preprocessing (ResNet default input size = 224)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    img_bytes = file.read()
    try:
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": "Invalid image file", "detail": str(e)}), 400

    save_path = os.path.join(IMAGE_SAVE_DIR, file.filename)
    try:
        with open(save_path, "wb") as f:
            f.write(img_bytes)
    except Exception:
        pass

    input_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).cpu().squeeze(0).tolist()
        best_idx = int(torch.argmax(logits, dim=1).item())
        best_prob = probs[best_idx]

    response = {
        "prediction": class_names[best_idx],
        "confidence": round(best_prob * 100, 2),
        "all_probs": {class_names[i]: round(p * 100, 2) for i, p in enumerate(probs)},
        "filename": file.filename
    }
    return jsonify(response)

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(IMAGE_SAVE_DIR, filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
