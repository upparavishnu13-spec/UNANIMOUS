#!/usr/bin/env python3
"""CropGuard AI - PlantVillage Dataset Downloader and Verifier

Helper utility to download, extract, and verify the PlantVillage dataset.
Supports:
  1. Kaggle API automated download (abdallahalidev/plantvillage-dataset)
  2. Direct URL download and extraction
  3. Dataset structure validation and class distribution report
"""

import argparse
import logging
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('download_dataset')

DEFAULT_KAGGLE_DATASET = "abdallahalidev/plantvillage-dataset"
EXPECTED_PLANTVILLAGE_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download and verify the PlantVillage dataset for CropGuard AI"
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['kaggle', 'direct', 'verify'],
        default='kaggle',
        help="Download method ('kaggle', 'direct', or 'verify' existing directory)"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='dataset/PlantVillage',
        help="Target destination directory (default: dataset/PlantVillage)"
    )
    parser.add_argument(
        '--kaggle_dataset',
        type=str,
        default=DEFAULT_KAGGLE_DATASET,
        help=f"Kaggle dataset handle (default: {DEFAULT_KAGGLE_DATASET})"
    )
    parser.add_argument(
        '--url',
        type=str,
        default=None,
        help="Direct download URL for dataset zip file (used with --method direct)"
    )
    return parser.parse_args()


def check_kaggle_credentials() -> bool:
    """Check if Kaggle API credentials are configured."""
    kaggle_json_user = Path.home() / ".kaggle" / "kaggle.json"
    env_user = os.environ.get("KAGGLE_USERNAME")
    env_key = os.environ.get("KAGGLE_KEY")
    return kaggle_json_user.exists() or (bool(env_user) and bool(env_key))


def print_kaggle_instructions():
    """Print setup instructions for Kaggle API authentication."""
    logger.error("\n" + "=" * 70)
    logger.error("KAGGLE CREDENTIALS NOT FOUND")
    logger.error("=" * 70)
    logger.error("To download automatically using Kaggle API, follow these steps:")
    logger.error("1. Sign in to your Kaggle account: https://www.kaggle.com")
    logger.error("2. Go to Account Settings: https://www.kaggle.com/settings")
    logger.error("3. Scroll to the 'API' section and click 'Create New Token'")
    logger.error("4. A file named 'kaggle.json' will be downloaded.")
    logger.error("5. Place it in:")
    if sys.platform == 'win32':
        logger.error(r"   C:\Users\<YourUsername>\.kaggle\kaggle.json")
    else:
        logger.error("   ~/.kaggle/kaggle.json")
    logger.error("\nAlternatively, set environment variables:")
    logger.error("   set KAGGLE_USERNAME=your_username")
    logger.error("   set KAGGLE_KEY=your_api_key")
    logger.error("\nOr manually download the dataset:")
    logger.error("   URL: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset")
    logger.error("   Extract the contents directly into: dataset/PlantVillage/")
    logger.error("=" * 70 + "\n")


def download_with_kaggle(dataset_handle: str, output_dir: Path) -> bool:
    """Download and extract dataset using official Kaggle API."""
    if not check_kaggle_credentials():
        print_kaggle_instructions()
        return False

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        logger.error("Kaggle authentication error: %s", e)
        print_kaggle_instructions()
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    temp_zip = output_dir.parent / "plantvillage_temp.zip"

    logger.info("Downloading dataset '%s' via Kaggle API...", dataset_handle)
    try:
        api.dataset_download_files(dataset_handle, path=str(output_dir.parent), unzip=False, quiet=False)
        
        # Identify downloaded zip file
        expected_zip = output_dir.parent / (dataset_handle.split('/')[-1] + ".zip")
        if expected_zip.exists():
            zip_target = expected_zip
        elif temp_zip.exists():
            zip_target = temp_zip
        else:
            found_zips = list(output_dir.parent.glob("*.zip"))
            if not found_zips:
                raise FileNotFoundError("Could not locate downloaded zip file.")
            zip_target = found_zips[0]

        logger.info("Extracting %s to %s...", zip_target.name, output_dir)
        with zipfile.ZipFile(zip_target, 'r') as zip_ref:
            zip_ref.extractall(output_dir)

        # Cleanup downloaded zip archive
        try:
            zip_target.unlink()
        except OSError:
            pass

        logger.info("Download and extraction complete.")
        return True
    except Exception as e:
        logger.error("Error during Kaggle download: %s", e)
        return False


def download_with_direct_url(url: str, output_dir: Path) -> bool:
    """Download a zip file from a direct URL with progress tracking."""
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_zip = output_dir.parent / "plantvillage_direct.zip"

    logger.info("Downloading from URL: %s", url)
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            downloaded = 0
            with open(temp_zip, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192 * 16):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            sys.stdout.write(f"\rDownloading: {downloaded / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB ({percent:.1f}%)")
                            sys.stdout.flush()
        print()
        logger.info("Extracting %s to %s...", temp_zip.name, output_dir)
        with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
            zip_ref.extractall(output_dir)

        try:
            temp_zip.unlink()
        except OSError:
            pass
        return True
    except Exception as e:
        logger.error("Direct download failed: %s", e)
        return False


def verify_dataset(target_dir: Path) -> Tuple[bool, Dict[str, int]]:
    """Inspect dataset directory, check class coverage, and print distribution."""
    if not target_dir.exists():
        logger.error("Dataset directory does not exist: %s", target_dir)
        return False, {}

    # Check for nested subdirectories
    subdirs = [p for p in target_dir.iterdir() if p.is_dir()]
    active_dir = target_dir
    if len(subdirs) < 10:
        for candidate_name in ['color', 'PlantVillage', 'plantvillage']:
            cand = target_dir / candidate_name
            if cand.is_dir() and len([p for p in cand.iterdir() if p.is_dir()]) >= 10:
                active_dir = cand
                break

    class_subdirs = sorted([p for p in active_dir.iterdir() if p.is_dir()])
    valid_exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

    logger.info("\n" + "=" * 70)
    logger.info("VERIFYING PLANTVILLAGE DATASET STRUCTURE")
    logger.info("Path: %s", active_dir)
    logger.info("=" * 70)
    logger.info("%-48s | %s", "Class Name", "Image Count")
    logger.info("-" * 70)

    class_counts: Dict[str, int] = {}
    total_images = 0

    for c_dir in class_subdirs:
        count = sum(1 for f in c_dir.iterdir() if f.suffix.lower() in valid_exts)
        class_counts[c_dir.name] = count
        total_images += count
        logger.info("%-48s | %6d", c_dir.name, count)

    logger.info("-" * 70)
    logger.info("Total Classes Found: %d | Total Images: %d", len(class_counts), total_images)
    logger.info("=" * 70)

    # Check for missing classes against canonical 38 PlantVillage classes
    missing_classes = [c for c in EXPECTED_PLANTVILLAGE_CLASSES if c not in class_counts]
    if missing_classes:
        logger.warning("Note: %d expected canonical classes were not matched by directory name:", len(missing_classes))
        for m in missing_classes[:5]:
            logger.warning("  - %s", m)
        if len(missing_classes) > 5:
            logger.warning("  ... and %d more", len(missing_classes) - 5)
    else:
        logger.info("All 38 canonical PlantVillage classes are present!")

    is_valid = len(class_counts) >= 10 and total_images > 100
    if is_valid:
        logger.info("Dataset verification PASSED. Dataset is ready for training.")
    else:
        logger.warning("Dataset verification FAILED. Found too few classes or images.")

    return is_valid, class_counts


def main():
    """Main CLI entrypoint."""
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()

    if args.method == 'verify':
        verify_dataset(output_dir)
        return

    if args.method == 'direct':
        if not args.url:
            logger.error("--url is required when using --method direct.")
            sys.exit(1)
        success = download_with_direct_url(args.url, output_dir)
    elif args.method == 'kaggle':
        success = download_with_kaggle(args.kaggle_dataset, output_dir)
    else:
        success = False

    if success:
        verify_dataset(output_dir)
    else:
        logger.error("Dataset acquisition failed.")
        sys.exit(1)


if __name__ == '__main__':
    main()
