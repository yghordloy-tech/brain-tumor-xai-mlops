"""
src/train.py
============
Training script for the Brain Tumor Classification project.
"""

import os
import pathlib

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import yaml

from src.data.loader import get_dataloaders
from src.models.classifier import get_model

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG = _PROJECT_ROOT / "configs" / "config.yaml"


def load_config(config_path=None):
    """Load configuration from a YAML file."""
    path = pathlib.Path(config_path) if config_path else _DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_device(device_cfg: str) -> torch.device:
    """Resolve the device string from config ('auto', 'cpu', or 'cuda')."""
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def main():
    cfg = load_config()

    seed = cfg["project"]["seed"]
    torch.manual_seed(seed)

    epochs = cfg["training"]["epochs"]
    learning_rate = cfg["training"]["learning_rate"]
    weight_decay = cfg["training"]["weight_decay"]
    save_dir = cfg["paths"]["model_checkpoint_dir"]

    wandb.init(project=cfg["project"]["name"], config=cfg)

    device = _resolve_device(cfg["project"]["device"])
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, class_to_name = get_dataloaders()
    model = get_model()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    os.makedirs(save_dir, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 20)

        model.train()
        running_train_loss = 0.0
        train_samples = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * images.size(0)
            train_samples += images.size(0)

        epoch_train_loss = running_train_loss / train_samples

        model.eval()
        running_val_loss = 0.0
        val_correct = 0
        val_samples = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data).item()
                val_samples += images.size(0)

        epoch_val_loss = running_val_loss / val_samples
        epoch_val_acc = val_correct / val_samples

        print(f"Train Loss: {epoch_train_loss:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | "
              f"Val Acc: {epoch_val_acc:.4f}")

        wandb.log({
            "epoch": epoch,
            "Training Loss": epoch_train_loss,
            "Validation Loss": epoch_val_loss,
            "Validation Accuracy": epoch_val_acc
        })

        if epoch_val_acc > best_val_acc:
            print(f"Validation accuracy improved from {best_val_acc:.4f} to {epoch_val_acc:.4f}. Saving model...")
            best_val_acc = epoch_val_acc
            save_path = os.path.join(save_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)

    print("\nTraining complete.")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")

    wandb.finish()

if __name__ == "__main__":
    main()
