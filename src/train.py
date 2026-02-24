"""
src/train.py
============
Training script for the Brain Tumor Classification project.
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import pathlib

from src.data.loader import get_dataloaders
from src.models.classifier import get_model

def load_config(config_path="Config.yaml"):
    """Load configuration from a YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    # 1. Load the config.yaml file
    config_path = "Config.yaml"
    cfg = load_config(config_path)

    # Extract configurations
    epochs = cfg["training"]["epochs"]
    learning_rate = cfg["training"]["learning_rate"]
    weight_decay = cfg["training"]["weight_decay"]
    save_dir = cfg["training"]["save_dir"]
    
    # 2. Initialize Weights & Biases using wandb.init and log the config parameters
    wandb.init(project='brain-tumor-xai', config=cfg)

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 3. Initialize the dataloaders and the model
    # Passing config_path to ensure they use the correct configuration file
    train_loader, val_loader, test_loader, class_to_name = get_dataloaders(config_path)
    model = get_model(config_path)
    model = model.to(device)

    # 4. Set up the CrossEntropyLoss function and the Adam optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    best_val_acc = 0.0

    # 5. Standard PyTorch training and validation loop
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 20)
        
        # --- Training Phase ---
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
        
        # --- Validation Phase ---
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

        # 6. Log the metrics to wandb
        wandb.log({
            "epoch": epoch,
            "Training Loss": epoch_train_loss,
            "Validation Loss": epoch_val_loss,
            "Validation Accuracy": epoch_val_acc
        })

        # 7. Save the model state dictionary ONLY if the validation accuracy improves
        if epoch_val_acc > best_val_acc:
            print(f"Validation accuracy improved from {best_val_acc:.4f} to {epoch_val_acc:.4f}. Saving model...")
            best_val_acc = epoch_val_acc
            save_path = os.path.join(save_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)

    print("\nTraining complete.")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    
    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    main()
