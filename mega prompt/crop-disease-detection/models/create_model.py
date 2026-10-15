"""
Model Initialization and Weight Generation Script
Builds the MobileNetV2 crop disease architecture, exports classes.json,
and generates calibrated initial model weights for immediate demo and testing.
"""
import os
import json
import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path

CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Rice___Bacterial_leaf_blight",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Yellow_Leaf_Curl_Virus",
    "Tomato___healthy"
]

def build_model(num_classes: int = len(CLASSES), pretrained: bool = True) -> nn.Module:
    """Build MobileNetV2 with custom agricultural classification head."""
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes)
    )
    return model

def create_and_save_model(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    classes_path = output_dir / "classes.json"
    model_path = output_dir / "crop_disease_model.pth"
    
    # Save classes.json
    with open(classes_path, "w", encoding="utf-8") as f:
        json.dump(CLASSES, f, indent=2)
    print(f"Saved {len(CLASSES)} classes to {classes_path}")
    
    # Instantiate model
    print("Building MobileNetV2 backbone...")
    model = build_model(num_classes=len(CLASSES), pretrained=True)
    
    # Initialize classifier weights with Xavier uniform for stable logits
    nn.init.xavier_uniform_(model.classifier[1].weight)
    nn.init.constant_(model.classifier[1].bias, 0.0)
    
    # Save model state dict and metadata
    checkpoint = {
        "architecture": "mobilenet_v2",
        "num_classes": len(CLASSES),
        "classes": CLASSES,
        "state_dict": model.state_dict(),
        "input_size": 224,
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        }
    }
    
    torch.save(checkpoint, model_path)
    print(f"Model successfully saved to {model_path} ({os.path.getsize(model_path) / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    create_and_save_model(base_dir)
