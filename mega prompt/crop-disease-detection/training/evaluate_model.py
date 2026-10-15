#!/usr/bin/env python3
"""CropGuard AI - Model Evaluation Pipeline

Evaluates trained MobileNetV2 model on PlantVillage test dataset.
Generates:
  - Accuracy, Precision, Recall, and F1-score (per-class, macro, weighted)
  - Full Scikit-Learn Classification Report
  - High-resolution Seaborn Confusion Matrix Heatmap
  - Comprehensive JSON evaluation results file
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
import tensorflow as tf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('evaluate_model')

IMG_SIZE: Tuple[int, int] = (224, 224)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate CropGuard AI Model Performance on PlantVillage Dataset"
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='models/crop_disease_model.h5',
        help="Path to trained .h5 model file (default: models/crop_disease_model.h5)"
    )
    parser.add_argument(
        '--labels_path',
        type=str,
        default='models/class_labels.json',
        help="Path to class labels JSON (default: models/class_labels.json)"
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        default='dataset/PlantVillage',
        help="Path to PlantVillage dataset directory (default: dataset/PlantVillage)"
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help="Evaluation batch size (default: 32)"
    )
    parser.add_argument(
        '--confusion_matrix_output',
        type=str,
        default='models/confusion_matrix.png',
        help="Output filepath for confusion matrix image (default: models/confusion_matrix.png)"
    )
    parser.add_argument(
        '--results_output',
        type=str,
        default='models/evaluation_results.json',
        help="Output filepath for JSON evaluation metrics (default: models/evaluation_results.json)"
    )
    return parser.parse_args()


def resolve_dataset_directory(base_path: str) -> Path:
    """Find the directory containing class subfolders."""
    target_path = Path(base_path).resolve()
    if not target_path.exists():
        raise FileNotFoundError(
            f"Dataset directory '{base_path}' not found. Cannot evaluate model."
        )

    subdirs = [p for p in target_path.iterdir() if p.is_dir()]
    if len(subdirs) >= 10:
        return target_path

    for candidate_name in ['color', 'PlantVillage', 'plantvillage']:
        candidate = target_path / candidate_name
        if candidate.is_dir():
            nested = [p for p in candidate.iterdir() if p.is_dir()]
            if len(nested) >= 10:
                return candidate

    return target_path


def load_class_labels(labels_path: str) -> List[str]:
    """Load class label list from JSON file."""
    path = Path(labels_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Class labels file not found at '{labels_path}'. Please run training first."
        )
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict) and "classes" in data:
        return data["classes"]
    elif isinstance(data, list):
        return data
    raise ValueError(f"Unexpected format in class labels file: {labels_path}")


def load_test_dataset(dataset_dir: Path, batch_size: int = 32) -> Tuple[tf.data.Dataset, List[str]]:
    """Load the 10% hold-out test dataset consistent with training split."""
    val_test_ds = tf.keras.utils.image_dataset_from_directory(
        directory=dataset_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="int",
        shuffle=False
    )
    class_names = val_test_ds.class_names
    val_test_cardinality = tf.data.experimental.cardinality(val_test_ds).numpy()
    val_batches = max(1, val_test_cardinality // 2)

    # Take the second half of the validation split as the hold-out test dataset
    test_ds = val_test_ds.skip(val_batches)
    return test_ds, class_names


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: str
) -> None:
    """Generate and save publication-quality confusion matrix heatmap."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Format labels for readability (crop: disease)
    formatted_labels = [
        name.replace("___", " - ").replace("_", " ") for name in class_names
    ]

    plt.figure(figsize=(20, 16))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=formatted_labels,
        yticklabels=formatted_labels,
        cbar_kws={'label': 'Number of Predictions'}
    )
    plt.title('CropGuard AI - Confusion Matrix (PlantVillage Test Set)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Class', fontsize=13, fontweight='bold', labelpad=12)
    plt.ylabel('Ground Truth Class', fontsize=13, fontweight='bold', labelpad=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Confusion matrix heatmap saved to: %s", output_path)


def main():
    """Execute model evaluation and generate reports."""
    args = parse_args()

    logger.info("Starting CropGuard AI Model Evaluation")
    
    # 1. Verify model file
    if not os.path.exists(args.model_path):
        logger.error(
            "Model file not found at '%s'. "
            "Please train the model first using 'python training/train_model.py'.",
            args.model_path
        )
        sys.exit(1)

    # 2. Load class labels
    try:
        class_labels = load_class_labels(args.labels_path)
        logger.info("Loaded %d class labels from %s", len(class_labels), args.labels_path)
    except Exception as e:
        logger.error("Failed to load class labels: %s", e)
        sys.exit(1)

    # 3. Load trained model
    logger.info("Loading model from: %s", args.model_path)
    model = tf.keras.models.load_model(args.model_path)

    # 4. Load test dataset
    try:
        dataset_dir = resolve_dataset_directory(args.dataset_path)
        test_ds, ds_class_names = load_test_dataset(dataset_dir, batch_size=args.batch_size)
    except Exception as e:
        logger.error("Failed to load test dataset: %s", e)
        sys.exit(1)

    # 5. Collect predictions and ground truth labels
    logger.info("Generating predictions on hold-out test set...")
    y_true_list = []
    y_pred_list = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        predicted_classes = np.argmax(preds, axis=1)
        y_pred_list.extend(predicted_classes)
        y_true_list.extend(labels.numpy())

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    if len(y_true) == 0:
        logger.error("No test samples found in dataset.")
        sys.exit(1)

    # 6. Compute Evaluation Metrics
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    per_class_p, per_class_r, per_class_f1, per_class_support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(len(class_labels))), zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_labels))))
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_labels))),
        target_names=class_labels,
        output_dict=True,
        zero_division=0
    )
    report_str = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_labels))),
        target_names=class_labels,
        zero_division=0
    )

    # 7. Print Terminal Summary
    logger.info("\n" + "=" * 65)
    logger.info("CropGuard AI Evaluation Summary")
    logger.info("=" * 65)
    logger.info("Total Test Samples Evaluated: %d", len(y_true))
    logger.info("Overall Accuracy:             %.4f (%.2f%%)", accuracy, accuracy * 100)
    logger.info("Macro Precision:              %.4f", macro_p)
    logger.info("Macro Recall:                 %.4f", macro_r)
    logger.info("Macro F1-Score:               %.4f", macro_f1)
    logger.info("Weighted Precision:           %.4f", weighted_p)
    logger.info("Weighted Recall:              %.4f", weighted_r)
    logger.info("Weighted F1-Score:            %.4f", weighted_f1)
    logger.info("=" * 65)
    logger.info("\nFull Classification Report:\n%s", report_str)

    # 8. Save Confusion Matrix Plot
    plot_confusion_matrix(cm, class_labels, args.confusion_matrix_output)

    # 9. Save JSON Evaluation Results
    per_class_results = {}
    for idx, name in enumerate(class_labels):
        per_class_results[name] = {
            "precision": float(per_class_p[idx]),
            "recall": float(per_class_r[idx]),
            "f1_score": float(per_class_f1[idx]),
            "support": int(per_class_support[idx])
        }

    evaluation_data: Dict[str, Any] = {
        "summary": {
            "total_test_samples": int(len(y_true)),
            "accuracy": accuracy,
            "macro_precision": float(macro_p),
            "macro_recall": float(macro_r),
            "macro_f1_score": float(macro_f1),
            "weighted_precision": float(weighted_p),
            "weighted_recall": float(weighted_r),
            "weighted_f1_score": float(weighted_f1)
        },
        "per_class_metrics": per_class_results,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist()
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.results_output)), exist_ok=True)
    with open(args.results_output, 'w', encoding='utf-8') as f:
        json.dump(evaluation_data, f, indent=2)
    logger.info("Evaluation results saved to: %s", args.results_output)


if __name__ == '__main__':
    main()
