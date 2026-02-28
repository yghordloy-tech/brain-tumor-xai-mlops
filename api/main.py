import io
import pathlib

import torch
import yaml
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from torchvision import transforms
from torchvision.models import resnet50

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG = _PROJECT_ROOT / "configs" / "config.yaml"


def _load_config():
    with open(_DEFAULT_CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


cfg = _load_config()

app = FastAPI(title="Brain Tumor MRI Diagnostic API", version="1.0")

device_cfg = cfg["project"]["device"]
if device_cfg == "auto":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(device_cfg)

class_names = cfg["data"]["class_names"]
num_classes = cfg["model"]["num_classes"]
image_size = cfg["data"]["image_size"]
norm_cfg = cfg["augmentation"]["normalize"]

model = resnet50(num_classes=num_classes)

model_path = _PROJECT_ROOT / cfg["paths"]["model_checkpoint_dir"] / "best_model.pth"
raw_state_dict = torch.load(str(model_path), map_location=device, weights_only=True)

clean_state_dict = {k.replace("backbone.", ""): v for k, v in raw_state_dict.items()}
model.load_state_dict(clean_state_dict)
model.to(device)
model.eval()

eval_transforms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=norm_cfg["mean"], std=norm_cfg["std"]),
])


@app.get("/")
def home():
    return {"message": "Welcome to the Brain Tumor XAI API. The model is locked and loaded!"}


@app.post("/predict")
async def predict_mri(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    input_tensor = eval_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)

    diagnosis = class_names[predicted_idx.item()]

    return {
        "filename": file.filename,
        "diagnosis": diagnosis,
        "confidence": f"{confidence.item() * 100:.2f}%"
    }