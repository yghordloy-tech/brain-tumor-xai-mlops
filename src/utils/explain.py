"""
src/utils/explain.py
====================
Script to generate Grad-CAM explainability heatmaps for Brain Tumor MRI Classification.
"""

import argparse
import pathlib

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms

# Absolute import allows the script to work when running as a module (-m src.utils.explain)
from src.models.classifier import get_model


def load_config(config_path="Config.yaml"):
    """Load configuration from a YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_heatmap(image_path: str, config_path="Config.yaml"):
    """
    Generate a Grad-CAM heatmap for a given image using the trained model.
    """
    # Resolve project root assuming this script is in src/utils/
    project_root = pathlib.Path(__file__).resolve().parents[2]
    cfg = load_config(project_root / config_path)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model architecture
    model = get_model(project_root / config_path)
    
    # Load trained weights
    save_dir = cfg["training"]["save_dir"]
    weights_path = project_root / save_dir / "best_model.pth"
    
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found at {weights_path}. Have you trained the model yet?")
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    # Define the target layer for Grad-CAM
    # Our BrainTumorClassifier stores the ResNet50 in `self.backbone`
    target_layers = [model.backbone.layer4[-1]]

    # Prepare the image
    image = Image.open(image_path).convert('RGB')
    
    # Get image size and normalization stats from config
    # Fallback to standard ImageNet sizes and normalizations if not present
    img_size = cfg["data"].get("img_size", 224) 
    mean = cfg.get("augmentation", {}).get("normalize", {}).get("mean", [0.485, 0.456, 0.406])
    std = cfg.get("augmentation", {}).get("normalize", {}).get("std", [0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    input_tensor = transform(image).unsqueeze(0).to(device)

    # Convert original PIL image to numpy representation for overlay, 
    # Shape must be (H, W, 3), values in range [0, 1]
    rgb_img = np.array(image.resize((img_size, img_size))) / 255.0

    # Initialize Grad-CAM
    cam = GradCAM(model=model, target_layers=target_layers)

    # If targets is None, the highest scoring category (prediction) will be used
    targets = None 

    # Generate CAM
    print(f"Generating Grad-CAM heatmap for {image_path}...")
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    
    # In this example, grayscale_cam has only one image in the batch
    grayscale_cam = grayscale_cam[0, :]

    # Overlay heatmap on the original image
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # Save the output
    output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    input_filename = pathlib.Path(image_path).name
    output_path = output_dir / f"gradcam_{input_filename}"
    
    # Convert RGB to BGR for saving with cv2
    cv2.imwrite(str(output_path), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    print(f"Heatmap successfully saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Grad-CAM heatmaps for Brain Tumor MRI images.")
    parser.add_argument("--image", type=str, required=True, help="Path to the source MRI image.")
    args = parser.parse_args()
    
    # We execute from the project root usually, but absolute python paths will work wherever it is run from now
    generate_heatmap(args.image)
