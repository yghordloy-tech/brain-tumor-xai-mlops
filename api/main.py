import io
import torch
import yaml
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from torchvision import transforms
from torchvision.models import resnet50

# 1. Initialize the Web App
app = FastAPI(title="Brain Tumor MRI Diagnostic API", version="1.0")

# 2. Setup Device and Class Names
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

# 3. Load the Model Architecture and Weights
model = resnet50(num_classes=4)

# Load the raw weights dictionary (with weights_only=True to fix the security warning)
raw_state_dict = torch.load("models/saved_weights/best_model.pth", map_location=device, weights_only=True)

# Strip the "backbone." prefix from the keys so they match the standard ResNet
clean_state_dict = {k.replace("backbone.", ""): v for k, v in raw_state_dict.items()}

# Load the cleaned weights into the model
model.load_state_dict(clean_state_dict)
model.to(device)
model.eval() # Set to evaluation mode (turns off training features)

# 4. Recreate the Image Preprocessing (from your config)
eval_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.get("/")
def home():
    return {"message": "Welcome to the Brain Tumor XAI API. The model is locked and loaded!"}

@app.post("/predict")
async def predict_mri(file: UploadFile = File(...)):
    """Accepts an uploaded MRI image and returns the AI's diagnosis."""
    # Read the uploaded image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Preprocess the image and send it to the GPU
    input_tensor = eval_transforms(image).unsqueeze(0).to(device)
    
    # Let the model make a prediction!
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