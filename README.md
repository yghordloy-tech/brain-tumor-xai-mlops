# 🧠 Brain Tumor MRI Classification — XAI & MLOps Pipeline

> **Explainable AI + production-grade MLOps for 4-class brain tumor classification from MRI scans.**

---

## 📁 Project Structure

```
brain-tumor-xai-mlops/
│
├── configs/
│   └── config.yaml              # Master hyperparameter & path configuration
│
├── src/
│   ├── data/
│   │   └── loader.py            # DataLoader factory with augmentation & weighted sampling
│   │
│   ├── models/
│   │   └── classifier.py        # ResNet50-based model definition & factory
│   │
│   ├── utils/
│   │   └── explain.py           # Grad-CAM explainability heatmap generation
│   │
│   └── train.py                 # Training & validation loop with W&B logging
│
├── api/
│   └── main.py                  # FastAPI inference endpoint
│
├── data/                        # (git-ignored, DVC-tracked)
│   ├── raw/
│   │   ├── Training/            # Class sub-folders (glioma, meningioma, etc.)
│   │   └── Testing/
│   └── processed/
│
├── models/
│   └── saved_weights/           # (git-ignored) saved model checkpoints
│
├── .github/
│   └── workflows/               # CI/CD pipeline
│
├── Dockerfile                   # Production container for the API
├── requirements.txt             # Full training dependencies
├── requirements-api.txt         # Lightweight API-only dependencies
├── pyproject.toml               # Tool configuration
└── README.md
```

## ⚙️ Configuration

All settings are centralized in [`configs/config.yaml`](configs/config.yaml). Every module reads from this single source of truth:

| Section          | Controls                                          |
|------------------|---------------------------------------------------|
| `project`        | Name, seed, device (`auto`/`cpu`/`cuda`)          |
| `paths`          | Data directories, model checkpoint dir, logs      |
| `data`           | Image size, class names, num workers, pin memory  |
| `augmentation`   | Flip probabilities, rotation, normalization stats |
| `model`          | Architecture, pretrained flag, num classes         |
| `training`       | Batch size, epochs, learning rate, weight decay   |
| `api`            | Host, port, reload flag                           |

## 🚀 Quick Start

```bash
# Clone & enter the repo
git clone <repo-url> && cd brain-tumor-xai-mlops

# Create virtual environment
python -m venv .venv && .venv\Scripts\activate       # Windows
# python -m venv .venv && source .venv/bin/activate   # Linux / macOS

# Install dependencies
pip install -r requirements.txt

# Pull data with DVC
dvc pull

# Train the model (logs to Weights & Biases)
python -m src.train

# Generate Grad-CAM explanation
python -m src.utils.explain --image path/to/mri.jpg

# Launch the API
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## 🐳 Docker Deployment

```bash
docker build -t brain-tumor-api .
docker run -p 8000:8000 brain-tumor-api
```

## 🔬 API Endpoints

| Method | Endpoint   | Description                              |
|--------|------------|------------------------------------------|
| GET    | `/`        | Health check                             |
| POST   | `/predict` | Upload an MRI image, receive a diagnosis |

**Example:**
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@path/to/mri_scan.jpg"
```

## 📦 Tech Stack

| Layer              | Tools                                  |
|--------------------|----------------------------------------|
| Deep Learning      | PyTorch, TorchVision                   |
| Explainability     | Grad-CAM (pytorch-grad-cam)            |
| Experiment Tracking| Weights & Biases                       |
| Data Versioning    | DVC                                    |
| API Serving        | FastAPI, Uvicorn                       |
| Containerisation   | Docker                                 |

## 📄 License

MIT
