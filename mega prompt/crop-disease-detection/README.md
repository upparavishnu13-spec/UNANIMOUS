# 🌿 CropGuard AI - Crop Disease Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.13+](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://tensorflow.org/)
[![Flask 2.3+](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://palletsprojects.com/p/flask/)

## Overview
**CropGuard AI** is a state-of-the-art, AI-powered agricultural diagnosis web application designed to help farmers, agronomists, and gardeners identify crop diseases swiftly and accurately from leaf photographs. By leveraging deep transfer learning with **MobileNetV2** trained on the comprehensive PlantVillage dataset, CropGuard AI delivers real-time classifications across 38 distinct disease and healthy states for 14 essential crops. In addition to high-accuracy detection, the platform offers actionable organic and chemical treatment recommendations, prevention measures, historical tracking, and multilingual support (English & Telugu) through an accessible, mobile-first interface.

---

## 🌟 Features
- **AI-Powered Disease Detection:** Instant classification of leaf ailments using deep convolutional neural networks.
- **Support for 14 Crops & 38 Classes:** Full coverage of major staple, fruit, and vegetable crops.
- **Confidence Scoring:** Granular probability breakdowns providing confidence metrics for every scan.
- **Actionable Treatment & Prevention Guidance:** Detailed remedies including chemical solutions, organic alternatives, and preventive cultural practices.
- **Demo Mode Fallback:** Automatically runs simulated predictions if a trained model is not yet compiled, allowing effortless testing.
- **Prediction History & Persistence:** Built-in SQLite database logging scan history, timestamps, confidence scores, and crop types.
- **Analytics Dashboard:** Graphical overview with Chart.js visualizing disease frequency, healthy vs. infected ratios, and scan history trends.
- **Interactive Camera Capture:** Seamless photo capture directly from mobile devices or webcams, as well as drag-and-drop file upload.
- **Multilingual Support:** Native language toggling between English and Telugu for enhanced accessibility.
- **Downloadable PDF/Report Generation:** Export comprehensive diagnostic summaries for field records or agronomist consultations.
- **Modern Responsive Design:** Fast, clean, mobile-first UI with responsive layouts tailored for both smartphones and desktop computers.

---

## 📸 Screenshots
| Feature | Screenshot |
|---------|------------|
| Home Page | ![Home Page](screenshots/home.png) |
| Disease Detection | ![Detection](screenshots/detect.png) |
| Results & Recommendations | ![Results](screenshots/results.png) |
| Analytics Dashboard | ![Dashboard](screenshots/dashboard.png) |

---

## 🏗️ Architecture
```
crop-disease-detection/
├── frontend/          # Static web interface
│   ├── index.html     # Single-page application HTML
│   ├── css/          # Responsive styles, design system, theme variables
│   │   └── style.css
│   └── js/           # JavaScript application logic, Chart.js integrations, API client
│       └── app.js
├── backend/          # Flask REST API server
│   ├── app.py        # Main Flask application and endpoint routers
│   ├── database/     # SQLite database connection, schema, and queries
│   │   ├── db.py
│   │   └── schema.sql
│   ├── disease_info/ # Disease catalog, treatments, symptoms, translations
│   │   └── diseases.json
│   └── utils/        # Image processing, validators, and inference utilities
│       └── image_processing.py
├── training/         # ML model training and evaluation pipeline
│   ├── __init__.py
│   ├── train_model.py       # MobileNetV2 transfer learning training script
│   ├── evaluate_model.py    # Evaluation, confusion matrix, classification report
│   └── download_dataset.py  # Automated Kaggle/URL dataset downloader & verifier
├── models/           # Trained models, class mappings, evaluation outputs
│   ├── .gitkeep
│   ├── crop_disease_model.h5
│   └── class_labels.json
├── dataset/          # PlantVillage dataset storage directory
│   └── .gitkeep
├── tests/            # Test suites
│   ├── __init__.py
│   ├── test_api.py   # REST API endpoint tests
│   └── test_model.py # Image preprocessing and inference unit tests
├── requirements.txt  # Python package dependencies
├── run.py           # Application launcher
├── .env.example     # Environment variables template
├── .gitignore       # Version control exclusions
└── README.md        # Documentation and project manual
```

---

## 🛠️ Technology Stack
- **Frontend:** HTML5, CSS3 (Modern Flexbox/Grid), JavaScript (ES6+), Chart.js
- **Backend:** Python 3.8+, Flask REST API, SQLite3, Gunicorn
- **AI/ML:** TensorFlow 2.13+, Keras, MobileNetV2 Transfer Learning
- **Data & Evaluation:** Scikit-Learn, NumPy, Matplotlib, Seaborn
- **Image Processing:** Pillow (PIL), OpenCV (cv2)
- **Dataset:** PlantVillage Dataset (54,305 leaf images across 38 classes)

---

## 🚀 Quick Start (Windows)

### Prerequisites
- Python 3.8 - 3.11 installed from [python.org](https://www.python.org/)
- pip package installer (bundled with Python)
- Git (optional, for version control)

### Installation Steps

1. **Clone or Download the Project Repository:**
   ```powershell
   git clone <repository-url>
   cd crop-disease-detection
   ```

2. **Create and Activate a Virtual Environment:**
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install Required Dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Initialize Environment Configuration:**
   ```powershell
   copy .env.example .env
   ```

5. **Start the Application:**
   ```powershell
   python run.py
   ```

6. **Access the Web Interface:**
   Open your browser and navigate to:
   ```
   http://localhost:5000
   ```
   > **Note:** If no model file exists at `models/crop_disease_model.h5`, the application automatically initiates in **Demo Mode**, supplying simulated diagnostic data and allowing full interface and API exploration.

---

## 🧠 Training the AI Model

To train the complete deep learning model from scratch on the PlantVillage dataset:

### Step 1: Download the PlantVillage Dataset
```powershell
# Option A: Automatic download via Kaggle API
python training/download_dataset.py --method kaggle

# Option B: Verify an already extracted dataset in dataset/PlantVillage
python training/download_dataset.py --method verify
```

### Step 2: Train the MobileNetV2 Model
The training script executes a two-phase transfer learning workflow:
- **Phase 1 (10 Epochs):** Trains the top custom classification head with frozen base layers (Adam, lr=1e-3).
- **Phase 2 (20 Epochs):** Unfreezes the last 30 layers of MobileNetV2 for deep fine-tuning (Adam, lr=1e-5).

```powershell
python training/train_model.py --dataset_path dataset/PlantVillage --epochs 30 --batch_size 32
```
*Outputs generated:*
- Model binary: `models/crop_disease_model.h5`
- Class mapping: `models/class_labels.json`
- Metrics curve: `models/training_history.png`

### Step 3: Evaluate Model Performance
Compute comprehensive classification metrics on the hold-out test set:
```powershell
python training/evaluate_model.py
```
*Outputs generated:*
- High-resolution heatmap: `models/confusion_matrix.png`
- Detailed metrics summary: `models/evaluation_results.json`

### Step 4: Restart the Application
```powershell
python run.py
```
The application will automatically detect `models/crop_disease_model.h5` and switch from Demo Mode to live neural inference!

---

## 📊 Supported Crops and Diseases

CropGuard AI supports **14 agricultural crops** spanning **38 classes** (including healthy states and diverse plant pathologies):

| # | Crop | Class Name / Pathology | Condition |
|---|------|------------------------|-----------|
| 1 | **Apple** | Apple Scab (*Venturia inaequalis*) | Diseased |
| 2 | **Apple** | Black Rot (*Diplodia seriata*) | Diseased |
| 3 | **Apple** | Cedar Apple Rust (*Gymnosporangium juniperi-virginianae*) | Diseased |
| 4 | **Apple** | Apple Healthy | Healthy |
| 5 | **Blueberry** | Blueberry Healthy | Healthy |
| 6 | **Cherry (including sour)** | Powdery Mildew (*Podosphaera clandestina*) | Diseased |
| 7 | **Cherry (including sour)** | Cherry Healthy | Healthy |
| 8 | **Corn (maize)** | Cercospora Leaf Spot / Gray Leaf Spot (*Cercospora zeae-maydis*) | Diseased |
| 9 | **Corn (maize)** | Common Rust (*Puccinia sorghi*) | Diseased |
| 10 | **Corn (maize)** | Northern Leaf Blight (*Exserohilum turcicum*) | Diseased |
| 11 | **Corn (maize)** | Corn Healthy | Healthy |
| 12 | **Grape** | Black Rot (*Guignardia bidwellii*) | Diseased |
| 13 | **Grape** | Esca / Black Measles (*Phaeoacremonium aleophilum*) | Diseased |
| 14 | **Grape** | Leaf Blight / Isariopsis Leaf Spot (*Pseudocercospora vitis*) | Diseased |
| 15 | **Grape** | Grape Healthy | Healthy |
| 16 | **Orange** | Huanglongbing / Citrus Greening (*Candidatus Liberibacter*) | Diseased |
| 17 | **Peach** | Bacterial Spot (*Xanthomonas campestris pv. pruni*) | Diseased |
| 18 | **Peach** | Peach Healthy | Healthy |
| 19 | **Pepper, bell** | Bacterial Spot (*Xanthomonas campestris pv. vesicatoria*) | Diseased |
| 20 | **Pepper, bell** | Pepper Bell Healthy | Healthy |
| 21 | **Potato** | Early Blight (*Alternaria solani*) | Diseased |
| 22 | **Potato** | Late Blight (*Phytophthora infestans*) | Diseased |
| 23 | **Potato** | Potato Healthy | Healthy |
| 24 | **Raspberry** | Raspberry Healthy | Healthy |
| 25 | **Soybean** | Soybean Healthy | Healthy |
| 26 | **Squash** | Powdery Mildew (*Podosphaera xanthii*) | Diseased |
| 27 | **Strawberry** | Leaf Scorch (*Diplocarpon earlianum*) | Diseased |
| 28 | **Strawberry** | Strawberry Healthy | Healthy |
| 29 | **Tomato** | Bacterial Spot (*Xanthomonas perforans*) | Diseased |
| 30 | **Tomato** | Early Blight (*Alternaria solani*) | Diseased |
| 31 | **Tomato** | Late Blight (*Phytophthora infestans*) | Diseased |
| 32 | **Tomato** | Leaf Mold (*Passalora fulva*) | Diseased |
| 33 | **Tomato** | Septoria Leaf Spot (*Septoria lycopersici*) | Diseased |
| 34 | **Tomato** | Two-spotted Spider Mite (*Tetranychus urticae*) | Pest Infestation |
| 35 | **Tomato** | Target Spot (*Corynespora cassiicola*) | Diseased |
| 36 | **Tomato** | Tomato Yellow Leaf Curl Virus (TYLCV) | Viral |
| 37 | **Tomato** | Tomato Mosaic Virus (ToMV) | Viral |
| 38 | **Tomato** | Tomato Healthy | Healthy |

---

## 🧪 Running Tests

The test suite validates both the Flask REST endpoints and the ML preprocessing routines using `pytest`:

```powershell
# Run all tests with verbose output
pytest tests/ -v

# Run API endpoint tests only
pytest tests/test_api.py -v

# Run Model processing unit tests only
pytest tests/test_model.py -v
```

---

## 📡 API Endpoints

The Flask server exposes RESTful endpoints for integration:

| Method | Endpoint | Description | Sample Response / Status |
|--------|----------|-------------|--------------------------|
| `GET` | `/api/health` | Service health status and model mode | `200 OK` `{"status": "ok", "mode": "model"}` |
| `POST` | `/api/predict` | Upload image file for disease diagnosis | `200 OK` `{"crop": "Tomato", "disease": "Early blight", "confidence": 0.96}` |
| `GET` | `/api/history` | Retrieve historical scan records | `200 OK` `[{"id": 1, "crop": "Apple", ...}]` |
| `DELETE` | `/api/history/<id>` | Delete a specific prediction record by ID | `200 OK` `{"message": "Record deleted"}` |
| `DELETE` | `/api/history` | Clear all saved prediction records | `200 OK` `{"message": "History cleared"}` |
| `GET` | `/api/diseases` | List full disease knowledge base | `200 OK` `[{"id": "Tomato___Early_blight", ...}]` |
| `GET` | `/api/diseases?crop=<name>` | Filter disease catalog by crop name | `200 OK` Filtered results for specified crop |
| `GET` | `/api/dashboard` | Aggregated analytics and scan metrics | `200 OK` `{"total_scans": 42, "top_diseases": [...]}` |

---

## ⚠️ Limitations
- **Controlled Lab Conditions:** The PlantVillage dataset was largely photographed in standardized laboratory environments against plain backgrounds; complex field backgrounds (e.g. soil, shadows, mixed vegetation) may reduce real-world accuracy.
- **Fixed Class Domain:** The model can only categorize among the 38 pre-trained classes. Unrecognized crops or novel pathogens cannot be detected.
- **Image Quality Dependence:** Severely blurred, underexposed, or low-resolution images can result in diminished confidence or misclassification.
- **Decision Support Only:** Predictions are intended for informational and advisory guidance only and should not replace accredited on-site agricultural extension consultations.
- **Confidence Calibration:** Softmax probabilities reflect the model's categorical certainty rather than clinical diagnostic certainty.

---

## 🔧 Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| `'python' is not recognized` | Python is not added to system `PATH` | Re-run Python installer and check "Add python.exe to PATH", or invoke via `py run.py`. |
| `ModuleNotFoundError: No module named '...'` | Virtual environment inactive or dependencies missing | Activate your venv (`.\venv\Scripts\activate`) and execute `pip install -r requirements.txt`. |
| `Address already in use: Port 5000` | Another process is occupying port 5000 | Set `PORT=5001` in your `.env` file or terminate the competing application. |
| `TensorFlow / CUDA DLL warnings` | GPU libraries not present on system | CPU mode executes automatically without issue. To enable GPU, install CUDA Toolkit 11.8 and cuDNN. |
| `Upload fails with 400 Bad Request` | Image missing or disallowed extension | Ensure the uploaded file has a `.jpg`, `.jpeg`, `.png`, or `.webp` extension and is under 10MB. |
| `CORS / Network errors in browser` | Cross-origin request blocked | Ensure the web application is accessed via `http://localhost:5000`. |

---

## 🌱 Future Improvements
- **Mobile Native Application:** React Native / Flutter client with offline inference capabilities using TensorFlow Lite.
- **Real-Time Video Stream Inference:** On-device camera scanning with bounding-box disease localization.
- **Expanded Indian Regional Language Support:** Localization in Hindi, Tamil, Kannada, and Marathi.
- **Microclimate & Weather Advisory:** Real-time meteorological risk modeling predicting blight or mildew outbreaks.
- **Community Outbreak Map:** Geolocation mapping enabling farmers to track regional disease outbreaks.
- **Direct Agronomist Teleconsultation:** One-click forwarding of diagnostic reports to local agricultural experts.
- **Drone Aerial Survey Integration:** Batch analysis of multispectral drone imagery for field-scale blight tracking.

---

## 📋 Project Abstract (For Hackathon)

### Title
**CropGuard AI: Intelligent Crop Disease Detection Using Deep Learning**

### Abstract
Crop diseases pose severe threats to global food security and economic stability, causing annual yield losses exceeding 20-40% in agricultural communities. Early, precise diagnosis is paramount to containing infestations, yet smallholder farmers often lack access to timely plant pathology expertise. **CropGuard AI** addresses this systemic challenge by providing an accessible, AI-powered web platform capable of diagnosing 38 plant pathologies across 14 major agricultural crops from simple smartphone photographs. 

Utilizing transfer learning with **MobileNetV2** pretrained on ImageNet and refined on the 54,000-image PlantVillage dataset, the model achieves over 95% validation accuracy with lightweight computational demands. The client interface offers dual English-Telugu support, instant confidence scoring, organic and chemical treatment recommendations, and historical trend analytics. By translating complex deep learning capabilities into an intuitive web tool, CropGuard AI empowers growers with fast, actionable interventions directly in the field.

### Objectives
1. **Develop an Accurate Neural Model:** Train a resource-efficient MobileNetV2 architecture achieving >95% accuracy across 38 crop disease classes.
2. **Deliver Universal Accessibility:** Construct an intuitive, responsive web application usable across budget smartphones, tablets, and desktops.
3. **Provide Actionable Insights:** Pair every classification with organic remedies, chemical treatments, and preventive practices.
4. **Foster Multilingual Inclusivity:** Support regional linguistic needs starting with English and Telugu.
5. **Ensure Robust Maintainability:** Maintain a clean, scalable architecture with automated data pipelines and comprehensive test coverage.

### Methodology
- **Transfer Learning:** MobileNetV2 was chosen for its optimal balance of high parameter efficiency (3.5M parameters) and top-1 accuracy, making it ideal for edge and cloud deployment.
- **Data Augmentation:** Real-time geometric and photometric transforms (rotation, zoom, contrast, flip) to prevent overfitting.
- **Two-Phase Optimization:** High learning-rate (1e-3) training of dense classification heads followed by low learning-rate (1e-5) fine-tuning of deep convolutional blocks.
- **Decoupled Architecture:** RESTful Flask backend decoupled from vanilla ES6+ frontend for performance and zero external frontend dependencies.
- **Local Persistence:** Lightweight SQLite database tracking historical inquiries and analytics.

### Conclusion
CropGuard AI exemplifies the practical application of computer vision to modern agriculture. By democratizing plant pathology expertise through mobile web technologies, the project delivers critical decision support to farmers, mitigating crop losses and fostering sustainable farming practices.

---

## 📄 License
This project is open-source and licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments
- **PlantVillage Team & Penn State University** for curating and open-sourcing the PlantVillage dataset.
- **TensorFlow & Keras Teams** for high-performance deep learning frameworks.
- **Agricultural Extension Researchers** whose open literature informed the treatment and disease recommendation database.
