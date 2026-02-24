# 🧠 Brain Tumor MRI Classification — XAI & MLOps Pipeline

> **Explainable AI + production-grade MLOps for 4-class brain tumor classification from MRI scans.**

---

## 📁 Project Structure

```
brain-tumor-xai-mlops/
│
├── configs/
│   └── config.yaml                   # Master hyperparameter & path configuration
│
├── src/                              # Core library
│   ├── data/
│   │   ├── dataset.py                # Custom PyTorch Dataset
│   │   ├── transforms.py            # Augmentation & normalization pipelines
│   │   ├── dataloader.py            # DataLoader factory
│   │   └── preprocessing.py         # Raw-data validation & preprocessing
│   │
│   ├── models/
│   │   ├── classifier.py            # Model factory (ResNet, EfficientNet, ViT)
│   │   ├── architectures.py         # Custom CNN / attention architectures
│   │   └── losses.py                # CE, Focal Loss, Label-smoothing CE
│   │
│   ├── training/
│   │   ├── trainer.py               # Training engine (AMP, grad-clip, logging)
│   │   ├── callbacks.py             # Early stopping & checkpoint callbacks
│   │   └── optimizers.py            # Optimizer & LR-scheduler factory
│   │
│   ├── evaluation/
│   │   ├── evaluator.py             # Test-set inference & metric computation
│   │   └── report.py               # Report generation (JSON / HTML)
│   │
│   ├── explainability/              # XAI methods
│   │   ├── gradcam.py               # GradCAM / GradCAM++ wrapper
│   │   ├── shap_explainer.py        # SHAP (DeepExplainer)
│   │   └── integrated_gradients.py  # Captum Integrated Gradients
│   │
│   └── utils/
│       ├── config.py                # YAML config loader
│       ├── logger.py                # Structured logger (Rich + W&B)
│       ├── metrics.py               # Accuracy, F1, AUC helpers
│       ├── helpers.py               # Seed, device, checkpoint I/O
│       └── visualization.py         # Plot curves, heatmaps, confusion matrix
│
├── api/                             # FastAPI serving layer
│   ├── main.py                      # App entry-point & startup
│   ├── routes.py                    # /predict, /predict/explain, /health
│   ├── schemas.py                   # Pydantic request / response models
│   └── inference.py                 # Model loading & prediction service
│
├── scripts/                         # CLI entry-points
│   ├── train.py                     # python scripts/train.py --config ...
│   ├── evaluate.py                  # python scripts/evaluate.py ...
│   ├── explain.py                   # python scripts/explain.py --image ...
│   └── export_model.py             # Export to ONNX / TorchScript
│
├── notebooks/                       # Jupyter-friendly exploration
│   ├── 01_eda.py
│   ├── 02_training_experiments.py
│   └── 03_xai_visualizations.py
│
├── tests/                           # pytest test suite
│   ├── test_data.py
│   ├── test_models.py
│   ├── test_api.py
│   └── test_explainability.py
│
├── data/                            # (git-ignored, DVC-tracked)
│   ├── raw/
│   └── processed/
│
├── checkpoints/                     # (git-ignored) saved model weights
├── outputs/                         # (git-ignored) explanations, reports
├── logs/                            # (git-ignored) training logs
│
├── .env.example                     # Environment variable template
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── dvc.yaml                         # DVC pipeline definition
├── pyproject.toml                   # Tool configuration (pytest, black, ruff, mypy)
├── requirements.txt                 # Python dependencies
└── README.md                        # ← You are here
```

## 🚀 Quick Start

```bash
# 1. Clone & enter the repo
git clone <repo-url> && cd brain-tumor-xai-mlops

# 2. Create virtual environment
python -m venv venv && venv\Scripts\activate   # Windows
# python -m venv venv && source venv/bin/activate  # Linux / macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy and edit environment variables
cp .env.example .env

# 5. Initialise DVC
dvc init

# 6. Train
python scripts/train.py --config configs/config.yaml

# 7. Evaluate
python scripts/evaluate.py --config configs/config.yaml --checkpoint checkpoints/best_model.pth

# 8. Generate explanations
python scripts/explain.py --config configs/config.yaml --checkpoint checkpoints/best_model.pth --image path/to/mri.jpg

# 9. Launch API
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## 🧪 Testing

```bash
pytest
```

## 📦 Tech Stack

| Layer              | Tools                                          |
| ------------------ | ---------------------------------------------- |
| Deep Learning      | PyTorch, TorchVision, TorchMetrics             |
| Explainability     | GradCAM, SHAP, Captum (Integrated Gradients)   |
| Experiment Tracking| Weights & Biases, MLflow                       |
| Data Versioning    | DVC                                            |
| API Serving        | FastAPI, Uvicorn                               |
| Containerisation   | Docker, Docker Compose                         |
| Code Quality       | Black, Ruff, MyPy, Pre-commit                  |

## 📄 License

MIT
