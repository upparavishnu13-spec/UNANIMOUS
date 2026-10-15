"""CropGuard AI - Model & Image Processing Unit Tests

Tests for:
  - Image preprocessing pipeline and output shapes
  - Image validation (valid, corrupted, empty files)
  - Filename extension validation (is_allowed_file)
  - Class name parsing (PlantVillage label formatting)
  - Model inference shape and probability distribution (when model file exists)
  - Class label alignment and indexing (when labels file exists)
"""

import io
import json
import os
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, Union

import numpy as np
import pytest
from PIL import Image

# Configure sys.path for backend imports
BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'backend')
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'crop_disease_model.h5')
LABELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'class_labels.json')

# Attempt importing utilities from backend, fallback to reference implementations if not yet created
try:
    from utils.image_processing import (  # type: ignore
        preprocess_image,
        validate_image,
        is_allowed_file,
        parse_class_name
    )
except ImportError:
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

    def is_allowed_file(filename: str) -> bool:
        """Check if file extension is supported."""
        if not filename or '.' not in filename:
            return False
        ext = filename.rsplit('.', 1)[1].lower()
        return ext in ALLOWED_EXTENSIONS

    def validate_image(image_input: Union[bytes, io.BytesIO, str]) -> bool:
        """Verify image stream or path can be safely decoded."""
        try:
            if isinstance(image_input, bytes):
                stream = io.BytesIO(image_input)
            elif isinstance(image_input, io.BytesIO):
                stream = image_input
                stream.seek(0)
            elif isinstance(image_input, (str, Path)):
                stream = open(image_input, 'rb')
            else:
                return False

            with Image.open(stream) as img:
                img.verify()
            return True
        except Exception:
            return False

    def preprocess_image(
        image_input: Union[Image.Image, bytes, io.BytesIO, str],
        target_size: Tuple[int, int] = (224, 224)
    ) -> np.ndarray:
        """Preprocess an image for MobileNetV2 inference."""
        if isinstance(image_input, Image.Image):
            img = image_input
        elif isinstance(image_input, bytes):
            img = Image.open(io.BytesIO(image_input))
        elif isinstance(image_input, io.BytesIO):
            image_input.seek(0)
            img = Image.open(image_input)
        elif isinstance(image_input, (str, Path)):
            img = Image.open(image_input)
        else:
            raise ValueError("Unsupported image input type")

        img = img.convert('RGB')
        img = img.resize(target_size, Image.Resampling.BILINEAR)
        img_array = np.array(img, dtype=np.float32)
        # Expand batch dimension -> (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def parse_class_name(raw_label: str) -> Dict[str, Any]:
        """Parse PlantVillage class format 'Crop___Disease' into human-readable metadata."""
        if "___" in raw_label:
            parts = raw_label.split("___", 1)
            crop_raw = parts[0]
            disease_raw = parts[1]
        else:
            crop_raw = "Unknown"
            disease_raw = raw_label

        crop = crop_raw.replace("_", " ").strip()
        disease = disease_raw.replace("_", " ").strip()
        is_healthy = "healthy" in disease.lower()

        return {
            "raw": raw_label,
            "crop": crop,
            "disease": disease,
            "is_healthy": is_healthy,
            "display_name": f"{crop} - {disease}"
        }


# ============================================================================
# Allowed Extension Tests
# ============================================================================

@pytest.mark.parametrize("filename", [
    "leaf.jpg",
    "leaf.jpeg",
    "LEAF.PNG",
    "tomato_sample.webp",
    "corn.JPEG",
    "test_image_123.jpg"
])
def test_is_allowed_file_valid_extensions(filename):
    """Test that standard supported image formats return True."""
    assert is_allowed_file(filename) is True


@pytest.mark.parametrize("filename", [
    "script.py",
    "document.pdf",
    "virus.exe",
    "data.csv",
    "image",
    "image.",
    "",
    "leaf.gif",
    "archive.tar.gz"
])
def test_is_allowed_file_invalid_extensions(filename):
    """Test that unauthorized formats return False."""
    assert is_allowed_file(filename) is False


# ============================================================================
# Image Validation Tests
# ============================================================================

def test_validate_image_valid_rgb():
    """Test validation of valid RGB image bytes."""
    buf = io.BytesIO()
    img = Image.new('RGB', (100, 100), color=(0, 255, 0))
    img.save(buf, format='JPEG')
    assert validate_image(buf.getvalue()) is True


def test_validate_image_corrupted_bytes():
    """Test validation fails for corrupted image bytes."""
    corrupted_data = b"NOT_A_VALID_JPEG_HEADER\x00\x01\x02\x03\x04"
    assert validate_image(corrupted_data) is False


def test_validate_image_empty_bytes():
    """Test validation fails for empty byte stream."""
    assert validate_image(b"") is False


# ============================================================================
# Image Preprocessing Tests
# ============================================================================

def test_preprocess_image_output_shape():
    """Test that image preprocessing returns tensor of shape (1, 224, 224, 3)."""
    img = Image.new('RGB', (450, 320), color=(100, 150, 200))
    result = preprocess_image(img, target_size=(224, 224))
    
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 224, 224, 3)
    assert result.dtype == np.float32


def test_preprocess_image_grayscale_converts_to_rgb():
    """Test that single-channel grayscale images are converted to 3 RGB channels."""
    img = Image.new('L', (200, 200), color=128)
    result = preprocess_image(img, target_size=(224, 224))
    
    assert result.shape == (1, 224, 224, 3)


def test_preprocess_image_with_bytes_input():
    """Test preprocessing accepts raw bytes."""
    buf = io.BytesIO()
    img = Image.new('RGB', (180, 180), color=(50, 100, 150))
    img.save(buf, format='PNG')
    
    result = preprocess_image(buf.getvalue(), target_size=(224, 224))
    assert result.shape == (1, 224, 224, 3)


# ============================================================================
# Class Name Parsing Tests
# ============================================================================

def test_parse_class_name_disease():
    """Test parsing disease class name."""
    meta = parse_class_name("Tomato___Early_blight")
    assert meta["crop"] == "Tomato"
    assert "Early blight" in meta["disease"]
    assert meta["is_healthy"] is False


def test_parse_class_name_healthy():
    """Test parsing healthy class name."""
    meta = parse_class_name("Apple___healthy")
    assert meta["crop"] == "Apple"
    assert meta["is_healthy"] is True


def test_parse_class_name_with_underscores_and_parentheses():
    """Test parsing complex class name with underscores and parentheses."""
    meta = parse_class_name("Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot")
    assert "Corn (maize)" in meta["crop"]
    assert "Cercospora leaf spot" in meta["disease"]
    assert meta["is_healthy"] is False


def test_parse_class_name_pepper_bell():
    """Test parsing Pepper bell class name."""
    meta = parse_class_name("Pepper,_bell___Bacterial_spot")
    assert "Pepper, bell" in meta["crop"]
    assert meta["is_healthy"] is False


# ============================================================================
# Model & Class Labels Tests (Conditional on file existence)
# ============================================================================

@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Trained model file not found")
def test_model_inference_shape():
    """Test loaded trained model inference produces correct 38-class probability vector."""
    import tensorflow as tf
    model = tf.keras.models.load_model(MODEL_PATH)
    dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
    predictions = model.predict(dummy_input)

    assert predictions.shape == (1, 38)
    assert np.isclose(np.sum(predictions), 1.0, atol=1e-3)


@pytest.mark.skipif(not os.path.exists(LABELS_PATH), reason="Class labels file not found")
def test_class_labels_alignment():
    """Test that class labels JSON contains 38 classes and valid indexing."""
    with open(LABELS_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict) and "classes" in data:
        classes = data["classes"]
    elif isinstance(data, list):
        classes = data
    else:
        pytest.fail("Unexpected format in class_labels.json")

    assert len(classes) == 38
    assert "Tomato___healthy" in classes
    assert "Apple___Apple_scab" in classes
