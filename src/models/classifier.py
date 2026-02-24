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


class BrainTumorClassifier(nn.Module):
    """
    ResNet50-based classifier for Brain Tumor MRI images.

    Parameters
    ----------
    num_classes : int
        The number of target classes (e.g., 4 for this dataset).
    pretrained : bool
        Whether to initialize with ImageNet pre-trained weights.
    """

    def __init__(self, num_classes: int = 4, pretrained: bool = True) -> None:
        super().__init__()

        # In newer torchvision versions, 'weights=ResNet50_Weights.DEFAULT'
        # is preferred, but 'pretrained' is still supported for compatibility.
        # We handle it by passing weights explicitly to avoid warnings if possible,
        # but the standard `pretrained` boolean argument works too.
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        
        # Initialize the base ResNet50 model
        self.backbone = models.resnet50(weights=weights)

        # Replace the final fully connected layer (model.fc)
        # ResNet50's default output size is 1000. It has an `fc` layer where:
        # in_features = 2048 (output of the final bottleneck layer)
        in_features = self.backbone.fc.in_features
        
        # Create a new Linear layer with `num_classes` outputs
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """Pass the input through the ResNet backbone."""
        return self.backbone(x)


def get_model(config_path: str | pathlib.Path | None = None) -> BrainTumorClassifier:
    """
    Initialize and return the BrainTumorClassifier based on config settings.

    Parameters
    ----------
    config_path : str or pathlib.Path, optional
        Path to the generic YAML configuration file. Defaults to `Config.yaml`
        at the root of the project.

    Returns
    -------
    BrainTumorClassifier
        The instantiated PyTorch model ready for training or inference.
    """
    # Use the root `Config.yaml` if no path is provided
    if config_path is None:
        project_root = pathlib.Path(__file__).resolve().parents[2]
        config_path = project_root / "Config.yaml"
    else:
        config_path = pathlib.Path(config_path)

    # Load configuration
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Extract model parameters from the YAML config
    num_classes = cfg.get("model", {}).get("num_classes", 4)
    pretrained = cfg.get("model", {}).get("pretrained", True)

    # Instantiate and return the model
    model = BrainTumorClassifier(num_classes=num_classes, pretrained=pretrained)
    
    return model
