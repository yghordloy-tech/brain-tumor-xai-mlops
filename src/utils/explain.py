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

from src.models.classifier import get_model

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _PROJECT_ROOT / "configs" / "config.yaml"


def _load_config(config_path=None):
    path = pathlib.Path(config_path) if config_path else _DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_heatmap(image_path: str, config_path=None):
    cfg = _load_config(config_path)

    device_cfg = cfg["project"]["device"]
    if device_cfg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_cfg)
    print(f"Using device: {device}")

    model = get_model(config_path)

    weights_path = _PROJECT_ROOT / cfg["paths"]["model_checkpoint_dir"] / "best_model.pth"

    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found at {weights_path}. Have you trained the model yet?")

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    target_layers = [model.backbone.layer4[-1]]

    image = Image.open(image_path).convert("RGB")

    img_size = cfg["data"]["image_size"]
    mean = cfg["augmentation"]["normalize"]["mean"]
    std = cfg["augmentation"]["normalize"]["std"]

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    input_tensor = transform(image).unsqueeze(0).to(device)

    rgb_img = np.array(image.resize((img_size, img_size))) / 255.0

    cam = GradCAM(model=model, target_layers=target_layers)

    print(f"Generating Grad-CAM heatmap for {image_path}...")
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)
    grayscale_cam = grayscale_cam[0, :]

    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    output_dir = _PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    input_filename = pathlib.Path(image_path).name
    output_path = output_dir / f"gradcam_{input_filename}"

    cv2.imwrite(str(output_path), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    print(f"Heatmap successfully saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Grad-CAM heatmaps for Brain Tumor MRI images.")
    parser.add_argument("--image", type=str, required=True, help="Path to the source MRI image.")
    args = parser.parse_args()

    generate_heatmap(args.image)
