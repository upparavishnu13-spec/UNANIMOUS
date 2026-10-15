#!/usr/bin/env python3
"""CropGuard AI - Model Training Pipeline

MobileNetV2 Transfer Learning for PlantVillage Crop Disease Classification.
Supports 38 classes across 14 crops with 2-phase training:
  Phase 1: Feature extraction (frozen base model)
  Phase 2: Fine-tuning top layers of base model
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('train_model')

# Model constants
IMG_SIZE: Tuple[int, int] = (224, 224)
INPUT_SHAPE: Tuple[int, int, int] = (224, 224, 3)
DEFAULT_NUM_CLASSES: int = 38
AUTOTUNE = tf.data.AUTOTUNE


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train CropGuard AI MobileNetV2 Disease Detection Model"
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        default='dataset/PlantVillage',
        help="Path to PlantVillage dataset directory containing class subfolders"
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help="Total training epochs across Phase 1 and Phase 2 (default: 30)"
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help="Batch size for training and validation (default: 32)"
    )
    parser.add_argument(
        '--model_output',
        type=str,
        default='models/crop_disease_model.h5',
        help="Target filepath for saved HDF5 model (default: models/crop_disease_model.h5)"
    )
    parser.add_argument(
        '--labels_output',
        type=str,
        default='models/class_labels.json',
        help="Target filepath for saved class labels JSON (default: models/class_labels.json)"
    )
    parser.add_argument(
        '--history_plot',
        type=str,
        default='models/training_history.png',
        help="Target filepath for training metrics plot (default: models/training_history.png)"
    )
    parser.add_argument(
        '--fine_tune',
        action='store_true',
        default=True,
        help="Whether to perform Phase 2 fine-tuning (default: True)"
    )
    parser.add_argument(
        '--no_fine_tune',
        action='store_false',
        dest='fine_tune',
        help="Disable Phase 2 fine-tuning"
    )
    return parser.parse_args()


def resolve_dataset_directory(base_path: str) -> Path:
    """Find the directory containing class subfolders in the given dataset path."""
    target_path = Path(base_path).resolve()
    if not target_path.exists():
        raise FileNotFoundError(
            f"Dataset directory '{base_path}' does not exist. "
            "Please run 'python training/download_dataset.py' to obtain the dataset."
        )

    # Check if direct subdirectories are classes
    subdirs = [p for p in target_path.iterdir() if p.is_dir()]
    if len(subdirs) >= 10:
        return target_path

    # Check common nested paths (e.g. PlantVillage/color, PlantVillage/PlantVillage)
    for candidate_name in ['color', 'PlantVillage', 'plantvillage']:
        candidate = target_path / candidate_name
        if candidate.is_dir():
            nested_subdirs = [p for p in candidate.iterdir() if p.is_dir()]
            if len(nested_subdirs) >= 10:
                logger.info("Resolved nested dataset folder: %s", candidate)
                return candidate

    return target_path


def print_class_distribution(dataset_dir: Path) -> List[Tuple[str, int]]:
    """Log class names and image counts found in the dataset."""
    valid_exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    class_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
    
    distribution = []
    total_images = 0
    
    logger.info("=" * 65)
    logger.info("Dataset Class Distribution: %s", dataset_dir)
    logger.info("=" * 65)
    logger.info("%-45s | %s", "Class Name", "Image Count")
    logger.info("-" * 65)
    
    for c_dir in class_dirs:
        count = sum(1 for f in c_dir.iterdir() if f.suffix.lower() in valid_exts)
        distribution.append((c_dir.name, count))
        total_images += count
        logger.info("%-45s | %6d", c_dir.name, count)
        
    logger.info("-" * 65)
    logger.info("Total classes: %d | Total images: %d", len(distribution), total_images)
    logger.info("=" * 65)
    return distribution


def create_data_pipeline(
    dataset_dir: Path,
    batch_size: int = 32
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, List[str]]:
    """Build train, validation, and test datasets with 80/10/10 split."""
    logger.info("Loading dataset from: %s", dataset_dir)

    # 80% train, 20% validation+test
    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory=dataset_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=True
    )

    val_test_ds = tf.keras.utils.image_dataset_from_directory(
        directory=dataset_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False
    )

    class_names = train_ds.class_names
    logger.info("Loaded %d classes from dataset", len(class_names))

    # Split 20% validation+test equally into 10% val and 10% test
    val_test_cardinality = tf.data.experimental.cardinality(val_test_ds).numpy()
    val_batches = max(1, val_test_cardinality // 2)

    val_ds = val_test_ds.take(val_batches)
    test_ds = val_test_ds.skip(val_batches)

    # Performance optimization: cache and prefetch
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


def build_cropguard_model(num_classes: int) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """Construct MobileNetV2 architecture with specified dense classification head.
    
    Architecture:
      - MobileNetV2 base (pretrained on ImageNet, frozen initially)
      - GlobalAveragePooling2D
      - Dense 256 + BatchNormalization + ReLU + Dropout(0.5)
      - Dense 128 + BatchNormalization + ReLU + Dropout(0.3)
      - Dense 38 + Softmax
    """
    # Data Augmentation sequential pipeline
    augmentation_layers = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical", name="aug_flip"),
        layers.RandomRotation(0.15, name="aug_rotation"),
        layers.RandomZoom(0.15, name="aug_zoom"),
        layers.RandomContrast(0.1, name="aug_contrast"),
        layers.RandomBrightness(0.1, name="aug_brightness"),
    ], name="data_augmentation")

    inputs = layers.Input(shape=INPUT_SHAPE, name="input_image")
    
    # Augmentation applied during training only
    x = augmentation_layers(inputs)
    
    # MobileNetV2 preprocessing (scales pixel values to [-1, 1])
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    
    # Base MobileNetV2 feature extractor
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=INPUT_SHAPE,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    
    # First classification block
    x = layers.Dense(256, use_bias=False, name="dense_256")(x)
    x = layers.BatchNormalization(name="bn_256")(x)
    x = layers.Activation('relu', name="relu_256")(x)
    x = layers.Dropout(0.5, name="dropout_256")(x)
    
    # Second classification block
    x = layers.Dense(128, use_bias=False, name="dense_128")(x)
    x = layers.BatchNormalization(name="bn_128")(x)
    x = layers.Activation('relu', name="relu_128")(x)
    x = layers.Dropout(0.3, name="dropout_128")(x)
    
    # Output classification layer
    outputs = layers.Dense(num_classes, activation='softmax', name="disease_prediction")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="CropGuard_MobileNetV2")
    return model, base_model


def plot_and_save_history(
    histories: List[tf.keras.callbacks.History],
    output_path: str
) -> None:
    """Plot training and validation accuracy and loss across phases."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    acc, val_acc, loss, val_loss = [], [], [], []
    for h in histories:
        acc.extend(h.history.get('accuracy', []))
        val_acc.extend(h.history.get('val_accuracy', []))
        loss.extend(h.history.get('loss', []))
        val_loss.extend(h.history.get('val_loss', []))

    epochs_range = range(1, len(acc) + 1)
    
    plt.figure(figsize=(14, 5))
    
    # Accuracy subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy', color='#2ecc71', lw=2)
    plt.plot(epochs_range, val_acc, label='Validation Accuracy', color='#27ae60', lw=2, linestyle='--')
    plt.title('Training and Validation Accuracy', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # Loss subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss', color='#e74c3c', lw=2)
    plt.plot(epochs_range, val_loss, label='Validation Loss', color='#c0392b', lw=2, linestyle='--')
    plt.title('Training and Validation Loss', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Training history plot saved to: %s", output_path)


def save_class_labels(class_names: List[str], output_path: str) -> None:
    """Save mapping of class indices and names to JSON format."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    mapping = {
        "classes": class_names,
        "class_to_idx": {name: idx for idx, name in enumerate(class_names)},
        "idx_to_class": {str(idx): name for idx, name in enumerate(class_names)},
        "num_classes": len(class_names)
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2)
    logger.info("Saved class labels (%d classes) to: %s", len(class_names), output_path)


def main():
    """Execute model training pipeline."""
    args = parse_args()
    
    logger.info("Starting CropGuard AI Model Training Pipeline")
    logger.info("TensorFlow Version: %s", tf.__version__)
    logger.info("Available GPUs: %s", tf.config.list_physical_devices('GPU'))

    # Ensure output directories exist
    os.makedirs(os.path.dirname(os.path.abspath(args.model_output)), exist_ok=True)

    # 1. Resolve and verify dataset
    dataset_dir = resolve_dataset_directory(args.dataset_path)
    print_class_distribution(dataset_dir)

    # 2. Build data pipelines
    train_ds, val_ds, test_ds, class_names = create_data_pipeline(
        dataset_dir=dataset_dir,
        batch_size=args.batch_size
    )
    num_classes = len(class_names)

    # 3. Save class labels
    save_class_labels(class_names, args.labels_output)

    # 4. Construct model
    model, base_model = build_cropguard_model(num_classes=num_classes)
    logger.info("\nCropGuard Model Architecture Summary:")
    model.summary(print_fn=logger.info)

    # Callbacks configuration
    checkpoint_cb = callbacks.ModelCheckpoint(
        filepath=args.model_output,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    early_stopping_cb = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr_cb = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )

    all_callbacks = [checkpoint_cb, early_stopping_cb, reduce_lr_cb]
    histories = []

    # ----------------------------------------------------
    # Phase 1: Train Top Layers Only
    # ----------------------------------------------------
    phase1_epochs = min(10, args.epochs)
    logger.info("\n" + "=" * 65)
    logger.info("PHASE 1: Training Classification Head (%d epochs, Adam lr=1e-3)", phase1_epochs)
    logger.info("=" * 65)
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )

    history_phase1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=phase1_epochs,
        callbacks=all_callbacks
    )
    histories.append(history_phase1)

    # ----------------------------------------------------
    # Phase 2: Fine-Tuning Top 30 Layers of Base Model
    # ----------------------------------------------------
    if args.fine_tune and (args.epochs > phase1_epochs):
        phase2_epochs = args.epochs - phase1_epochs
        logger.info("\n" + "=" * 65)
        logger.info("PHASE 2: Fine-tuning last 30 layers (%d epochs, Adam lr=1e-5)", phase2_epochs)
        logger.info("=" * 65)

        base_model.trainable = True
        # Freeze all layers except the last 30
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        for layer in base_model.layers[-30:]:
            layer.trainable = True

        logger.info("Trainable weights count: %d", len(model.trainable_weights))

        model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
        )

        history_phase2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            initial_epoch=phase1_epochs,
            callbacks=all_callbacks
        )
        histories.append(history_phase2)

    # 5. Evaluate on Hold-out Test Set
    logger.info("\n" + "=" * 65)
    logger.info("Evaluating Best Model on Hold-out Test Set")
    logger.info("=" * 65)
    
    if os.path.exists(args.model_output):
        best_model = tf.keras.models.load_model(args.model_output)
    else:
        best_model = model

    test_results = best_model.evaluate(test_ds, verbose=1)
    for name, value in zip(best_model.metrics_names, test_results):
        logger.info("Test %s: %.4f", name, value)

    # 6. Save Training History Visualizations
    plot_and_save_history(histories, args.history_plot)

    logger.info("\nTraining pipeline completed successfully.")
    logger.info("Best model saved at: %s", args.model_output)
    logger.info("Class labels saved at: %s", args.labels_output)


if __name__ == '__main__':
    main()
