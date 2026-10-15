# PlantVillage Dataset Download & Setup Guide

This project supports the standard **PlantVillage Dataset**, one of the most widely used and benchmarked plant disease image datasets in agricultural machine learning.

---

## 1. Dataset Overview
- **Origin**: Penn State University / EPFL (Crowdsourced Open Agriculture Data).
- **Classes**: 38 classes of healthy and diseased plant leaves across 14 crop species.
- **Image Count**: 54,303 individual leaf images captured under laboratory conditions against uniform backgrounds.
- **License**: Public domain / CC-BY (Open Data Commons).

Our default configuration focuses on **18 high-priority classes** across 6 major agricultural staples:
1. Apple (Apple Scab, Black Rot, Cedar Apple Rust, Healthy)
2. Corn/Maize (Common Rust, Northern Leaf Blight, Healthy)
3. Potato (Early Blight, Late Blight, Healthy)
4. Tomato (Bacterial Spot, Early Blight, Late Blight, Yellow Leaf Curl Virus, Healthy)
5. Bell Pepper (Bacterial Spot, Healthy)
6. Rice (Bacterial Leaf Blight)

---

## 2. Download Options

### Option A: Direct Download via Kaggle CLI (Recommended)
1. Ensure your Kaggle API key (`kaggle.json`) is placed in `~/.kaggle/` (Windows: `C:\Users\<Username>\.kaggle\kaggle.json`).
2. Run the Kaggle CLI command:
```bash
kaggle datasets download -d emmarex/plantdisease -p dataset/raw/ --unzip
```
Or use the New Plant Diseases Dataset (87K augmented):
```bash
kaggle datasets download -d vipoooool/new-plant-diseases-dataset -p dataset/raw/ --unzip
```

### Option B: Web Browser Download
1. Visit [Kaggle Plant Disease Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease).
2. Click **Download** (approx. 800 MB - 1.5 GB).
3. Extract the contents into `crop-disease-detection/dataset/raw/`.

### Option C: GitHub Open Source Mirror
Clone or download the raw images from the official repository:
```bash
git clone https://github.com/spMohanty/PlantVillage-Dataset.git dataset/raw/
```

---

## 3. Directory Layout for Training
Organize the dataset into standard `train/`, `val/`, and `test/` splits (80% / 10% / 10%):

```text
crop-disease-detection/dataset/
├── train/
│   ├── Apple___Apple_scab/
│   ├── Apple___Black_rot/
│   ├── Tomato___Early_blight/
│   └── ...
├── val/
│   ├── Apple___Apple_scab/
│   ├── ...
├── test/
│   ├── Apple___Apple_scab/
│   └── ...
└── sample_images/          # Built-in demonstration samples
    ├── tomato_early_blight.jpg
    ├── potato_late_blight.jpg
    ├── corn_common_rust.jpg
    ├── apple_scab.jpg
    ├── tomato_healthy.jpg
    └── non_leaf_sample.jpg
```

---

## 4. Running the Training Pipeline
Once downloaded and arranged, train the MobileNetV2 architecture by running:

```bash
cd crop-disease-detection
python training/train.py --data_dir dataset/ --epochs 15 --batch_size 32 --lr 0.0005
```

Evaluate the model and produce confusion matrix & classification metrics:
```bash
python training/evaluate.py --data_dir dataset/test/ --model_path models/crop_disease_model.pth
```

---

## 5. Important Model Limitations & Real-World Notice
- **Laboratory vs. Field Conditions**: PlantVillage images were captured under controlled lighting with uniform backgrounds. Real-world field images may feature shadows, complex soil backgrounds, overlapping weeds, or multiple co-occurring deficiencies.
- **Advisory Role**: The AI model assists in screening and early detection. High-stakes crop decisions should always be confirmed by an agricultural extension specialist or agronomist.
