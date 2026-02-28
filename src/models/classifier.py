"""
src/models/classifier.py
========================
PyTorch model definitions for the Brain Tumor Classification project.
"""

from __future__ import annotations

import pathlib

import torch.nn as nn
import torchvision.models as models
import yaml

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _PROJECT_ROOT / "configs" / "config.yaml"


def _load_config(config_path=None):
    """Load and return the YAML configuration dictionary."""
    path = pathlib.Path(config_path) if config_path else _DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class BrainTumorClassifier(nn.Module):
    """ResNet50-based classifier for Brain Tumor MRI images."""

    def __init__(self, num_classes: int = 4, pretrained: bool = True) -> None:
        super().__init__()

        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


def get_model(config_path=None) -> BrainTumorClassifier:
    """Initialize and return the BrainTumorClassifier based on config settings."""
    cfg = _load_config(config_path)

    num_classes = cfg["model"]["num_classes"]
    pretrained = cfg["model"]["pretrained"]

    return BrainTumorClassifier(num_classes=num_classes, pretrained=pretrained)
