"""Image processing utilities for CropGuard AI.

Handles image validation, metadata extraction, format checking, and model preprocessing.
"""

import logging
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
MIN_IMAGE_DIMENSION = 32


def is_allowed_file(filename: str) -> bool:
    """Check if the provided filename has an allowed image extension.

    Args:
        filename: Name or path of the uploaded file.

    Returns:
        True if the file extension is one of {'jpg', 'jpeg', 'png', 'webp'}, False otherwise.
    """
    if not filename or "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower().strip()
    return ext in ALLOWED_EXTENSIONS


def validate_image(file_path: str) -> Tuple[bool, str]:
    """Validate that the file exists, is a readable image, and meets minimum dimension requirements.

    Args:
        file_path: Absolute or relative path to the image file.

    Returns:
        A tuple of (is_valid: bool, message: str).
    """
    if not os.path.exists(file_path):
        return False, "File does not exist."

    if os.path.getsize(file_path) == 0:
        return False, "File is empty."

    try:
        with Image.open(file_path) as img:
            # Attempt to verify integrity
            img.verify()

        # Reopen to test loading pixel data and verify dimensions
        with Image.open(file_path) as img:
            img.load()
            width, height = img.size
            if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
                return (
                    False,
                    f"Image resolution too small ({width}x{height}). Minimum required is {MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION}.",
                )

        return True, "Valid"

    except UnidentifiedImageError:
        logger.warning("Unidentified image error encountered for file: %s", file_path)
        return False, "The uploaded file is not a recognized image format."
    except Exception as e:
        logger.warning("Image validation failed for %s: %s", file_path, str(e))
        return False, f"Invalid or corrupted image file: {str(e)}"


def get_image_info(file_path: str) -> Dict[str, Any]:
    """Extract metadata and details from an image file.

    Args:
        file_path: Path to the image file.

    Returns:
        Dictionary containing width, height, format, mode, and file_size in bytes.
    """
    info = {
        "width": 0,
        "height": 0,
        "format": "UNKNOWN",
        "mode": "UNKNOWN",
        "file_size": 0,
    }

    if not os.path.exists(file_path):
        return info

    try:
        info["file_size"] = os.path.getsize(file_path)
        with Image.open(file_path) as img:
            info["width"], info["height"] = img.size
            info["format"] = img.format or "UNKNOWN"
            info["mode"] = img.mode or "UNKNOWN"
    except Exception as e:
        logger.warning("Failed to extract image info for %s: %s", file_path, str(e))

    return info


def preprocess_image(
    image_path: str, target_size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    """Preprocess an image for neural network inference.

    Steps:
    1. Open image with PIL
    2. Convert to RGB mode
    3. Resize to target_size (224x224)
    4. Convert to numpy array with float32 type
    5. Normalize pixel values to [0, 1] (divide by 255.0)
    6. Expand dimensions to shape (1, height, width, 3)

    Args:
        image_path: Path to the image file.
        target_size: Desired (width, height) dimensions.

    Returns:
        Numpy array of shape (1, 224, 224, 3) normalized to [0, 1].
    """
    try:
        # Determine appropriate resampling filter for Pillow version
        if hasattr(Image, "Resampling"):
            resample_filter = Image.Resampling.BILINEAR
        elif hasattr(Image, "BILINEAR"):
            resample_filter = Image.BILINEAR
        else:
            resample_filter = getattr(Image, "LANCZOS", 1)

        with Image.open(image_path) as img:
            # Convert to RGB (handles RGBA, Grayscale, CMYK, etc.)
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Resize to target size
            resized_img = img.resize(target_size, resample=resample_filter)

            # Convert to float32 numpy array
            img_array = np.asarray(resized_img, dtype=np.float32)

            # Normalize to [0, 1]
            img_array = img_array / 255.0

            # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
            batch_array = np.expand_dims(img_array, axis=0)

            return batch_array

    except Exception as e:
        logger.error("Error preprocessing image %s: %s", image_path, str(e))
        raise RuntimeError(f"Image preprocessing failed: {str(e)}") from e
