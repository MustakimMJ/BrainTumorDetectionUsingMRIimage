# Brain Tumor Detector — Production Deployment

Flask web application for MRI brain tumor classification using a PyTorch ResNet-18 model. This project is configured for production deployment with Docker, Gunicorn, and Hugging Face model downloads.

## Project Structure

```
.
├── app/
│   ├── __init__.py      # Application factory, logging, CORS, error handlers
│   ├── config.py        # Environment-based configuration
│   ├── model.py         # Hugging Face download + model initialization
│   └── routes.py        # HTTP routes (/ , /predict, /health, /uploads)
├── templates/           # HTML templates (unchanged)
├── static/              # Static assets (ROC curve, confusion matrix)
├── labels.txt           # Class labels
├── uploads/             # Uploaded images (created at runtime)
├── models/              # Downloaded model weights (created at runtime)
├── app.py               # Local development entry point
├── wsgi.py              # Gunicorn entry point
├── gunicorn.conf.py     # Gunicorn configuration
├── requirements.txt     # Python dependencies
├── Dockerfile           # Production container image
├── .dockerignore
├── .gitignore
└── .env.example         # Example environment variables
```

## Prerequisites

- Python 3.11+
- Docker (optional, for containerized deployment)
- A Hugging Face account and model repository (for production)

Upload your trained `brain_tumor_model.pth` to a Hugging Face model repository before deploying.

---

## Local Execution

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Copy the example file and edit it:

```bash
cp .env.example .env
```

Set at minimum:

```env
HUGGINGFACE_REPO_ID=your-username/your-model-repo
HUGGINGFACE_MODEL_FILENAME=brain_tumor_model.pth
```

**Skip download for local development:** place the model file at `models/brain_tumor_model.pth`. If the file already exists, it will not be downloaded again.

### 4. Run the application

**Development (Flask built-in server):**

```bash
python app.py
```

**Production-like (Gunicorn, recommended):**

```bash
gunicorn --config gunicorn.conf.py wsgi:app
```

Open [http://localhost:5000](http://localhost:5000) (Flask dev) or [http://localhost:8080](http://localhost:8080) (Gunicorn default).

### 5. Verify health

```bash
curl http://localhost:8080/health
```

Expected response:

```json
{"status":"healthy"}
```

---

## Docker Build

Build the image from the project root:

```bash
docker build -t brain-tumor-detector .
```

The image uses:

- `python:3.11-slim` for a smaller footprint
- CPU-only PyTorch wheels
- A non-root `app` user
- Built-in `/health` check

Model weights are **not** baked into the image. They are downloaded at container startup from Hugging Face (unless already cached in the container volume).

---

## Docker Run

```bash
docker run --rm -p 8080:8080 \
  -e HUGGINGFACE_REPO_ID=your-username/your-model-repo \
  -e HUGGINGFACE_MODEL_FILENAME=brain_tumor_model.pth \
  brain-tumor-detector
```

### Persist model cache and uploads

```bash
docker run --rm -p 8080:8080 \
  -e HUGGINGFACE_REPO_ID=your-username/your-model-repo \
  -v brain-tumor-models:/app/models \
  -v brain-tumor-uploads:/app/uploads \
  brain-tumor-detector
```

On subsequent starts, the model is reused from `/app/models` and **not** downloaded again.

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HUGGINGFACE_REPO_ID` | Yes* | — | Hugging Face repository (e.g. `user/brain-tumor-model`) |
| `HUGGINGFACE_MODEL_FILENAME` | No | `brain_tumor_model.pth` | Model filename inside the repo |
| `HUGGINGFACE_TOKEN` | No | — | Token for private Hugging Face repos |
| `MODEL_DIR` | No | `./models` | Local directory for model weights |
| `UPLOAD_DIR` | No | `./uploads` | Directory for uploaded images |
| `LABELS_PATH` | No | `./labels.txt` | Path to class labels file |
| `PORT` | No | `8080` | HTTP port (used by Gunicorn and cloud platforms) |
| `LOG_LEVEL` | No | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `CORS_ORIGINS` | No | `*` | Comma-separated allowed origins, or `*` for all |
| `WEB_CONCURRENCY` | No | `1` | Gunicorn worker count (keep at `1` for ML inference) |
| `GUNICORN_TIMEOUT` | No | `120` | Request timeout in seconds |
| `FLASK_DEBUG` | No | `0` | Set to `1` for Flask debug mode (local only) |

\*Not required if the model file already exists in `MODEL_DIR`.

---

## Hugging Face Model Download

1. Upload your `.pth` file to a Hugging Face **model** repository.
2. Set `HUGGINGFACE_REPO_ID` and `HUGGINGFACE_MODEL_FILENAME`.
3. On startup, the app checks whether the file exists at `MODEL_DIR/HUGGINGFACE_MODEL_FILENAME`.
4. If the file is missing, it is downloaded via `huggingface_hub`.
5. If the file is present (e.g. from a previous run or mounted volume), download is skipped.

For private repositories, set `HUGGINGFACE_TOKEN` to a Hugging Face access token with read access.

---

## Deployment

All platforms below inject a `PORT` environment variable. The app listens on `0.0.0.0:$PORT` automatically.

### Render

1. Connect your Git repository.
2. Choose **Docker** as the environment (or **Python** with start command below).
3. Set environment variables (`HUGGINGFACE_REPO_ID`, etc.).
4. Render sets `PORT` automatically.

**Start command (non-Docker):**

```bash
gunicorn --config gunicorn.conf.py wsgi:app
```

### Railway

1. Create a new project from this repository.
2. Railway detects the `Dockerfile` automatically.
3. Add environment variables in the Railway dashboard.
4. Deploy — Railway assigns `PORT` at runtime.

Use the Railway skill or CLI:

```bash
railway up
```

### Google Cloud Run

```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/brain-tumor-detector
gcloud run deploy brain-tumor-detector \
  --image gcr.io/PROJECT_ID/brain-tumor-detector \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars HUGGINGFACE_REPO_ID=your-username/your-model-repo,HUGGINGFACE_MODEL_FILENAME=brain_tumor_model.pth \
  --memory 2Gi \
  --timeout 300
```

Cloud Run requires the container to listen on `$PORT` (default handled by Gunicorn config).

### General recommendations

- Allocate at least **2 GB RAM** (PyTorch + model weights).
- Keep `WEB_CONCURRENCY=1` to avoid loading the model in multiple workers.
- Mount persistent storage for `MODEL_DIR` if you want faster cold starts after the first download.
- Use `/health` for load balancer and platform health checks.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Web UI |
| `GET` | `/health` | Health check — returns `{"status":"healthy"}` |
| `POST` | `/predict` | Upload an image (`image` form field) and get prediction JSON |
| `GET` | `/uploads/<filename>` | Serve previously uploaded images |
| `GET` | `/static/<filename>` | Static assets |

---

## License

See project license. Model predictions are for research/educational use only and are not a substitute for professional medical diagnosis.
