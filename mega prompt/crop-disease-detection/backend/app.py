"""CropGuard AI - Backend Application

Flask REST API for crop disease detection, history tracking, disease information,
and dashboard metrics.
"""

import datetime
import json
import logging
import os
import random
import uuid
import sys
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

# Ensure project root and backend dir are in sys.path for direct script execution
BACKEND_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# Import local modules with relative/direct fallback
try:
    from backend.database.db import (
        delete_all_predictions,
        delete_prediction,
        get_all_predictions,
        get_dashboard_stats,
        get_prediction_by_id,
        init_db,
        save_prediction,
    )
    from backend.utils.image_processing import (
        is_allowed_file,
        preprocess_image,
        validate_image,
    )
except ImportError:
    from database.db import (
        delete_all_predictions,
        delete_prediction,
        get_all_predictions,
        get_dashboard_stats,
        get_prediction_by_id,
        init_db,
        save_prediction,
    )
    from utils.image_processing import (
        is_allowed_file,
        preprocess_image,
        validate_image,
    )

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Base directories
BACKEND_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
FRONTEND_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "frontend"))
UPLOADS_DIR = os.path.abspath(os.path.join(BACKEND_DIR, "uploads"))
DISEASES_JSON_PATH = os.path.abspath(
    os.path.join(BACKEND_DIR, "disease_info", "diseases.json")
)

# Global variables for model and disease knowledge
AI_MODEL = None
CLASS_LABELS: List[str] = []
DISEASES_DATA: Dict[str, Any] = {}
DEMO_MODE: bool = False


def load_disease_database() -> Dict[str, Any]:
    """Load the disease information database from JSON."""
    if os.path.exists(DISEASES_JSON_PATH):
        try:
            with open(DISEASES_JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                logger.info("Successfully loaded %d disease definitions.", len(data))
                return data
        except Exception as e:
            logger.error("Failed to parse %s: %s", DISEASES_JSON_PATH, str(e))
    else:
        logger.warning("Disease database not found at %s", DISEASES_JSON_PATH)
    return {}


def parse_class_name(class_name: str) -> Tuple[str, str, bool]:
    """Parse class name into Crop, Disease, and is_healthy flag.

    Examples:
        'Tomato___Late_blight' -> ('Tomato', 'Late Blight', False)
        'Tomato___healthy' -> ('Tomato', 'Healthy', True)
        'Corn_(maize)___Common_rust_' -> ('Corn (Maize)', 'Common Rust', False)
        'Cherry_(including_sour)___Powdery_mildew' -> ('Cherry', 'Powdery Mildew', False)
        'Pepper,_bell___Bacterial_spot' -> ('Pepper Bell', 'Bacterial Spot', False)
    """
    # Check if exact match exists in the disease database first
    if class_name in DISEASES_DATA:
        info = DISEASES_DATA[class_name]
        return info.get("crop", "Unknown"), info.get("disease", "Unknown"), info.get("is_healthy", False)

    # Dynamic parsing fallback
    parts = class_name.split("___")
    crop_raw = parts[0]
    disease_raw = parts[1] if len(parts) > 1 else ""

    # Map special crops
    if "Corn_(maize)" in crop_raw:
        crop = "Corn (Maize)"
    elif "Cherry_(including_sour)" in crop_raw:
        crop = "Cherry"
    elif "Pepper,_bell" in crop_raw:
        crop = "Pepper Bell"
    else:
        crop = crop_raw.replace("_", " ").strip()

    is_healthy = disease_raw.lower() == "healthy" or "healthy" in class_name.lower()

    if is_healthy:
        disease = "Healthy"
    else:
        # Handle special disease strings
        if "Cercospora" in disease_raw and "Gray_leaf_spot" in disease_raw:
            disease = "Cercospora Leaf Spot (Gray Leaf Spot)"
        elif "Spider_mites" in disease_raw:
            disease = "Spider Mites (Two-spotted Spider Mite)"
        elif "Tomato_Yellow_Leaf_Curl_Virus" in disease_raw:
            disease = "Tomato Yellow Leaf Curl Virus (TYLCV)"
        elif "Tomato_mosaic_virus" in disease_raw:
            disease = "Tomato Mosaic Virus (ToMV)"
        elif "Haunglongbing" in disease_raw:
            disease = "Huanglongbing (Citrus Greening)"
        elif "Esca" in disease_raw:
            disease = "Esca (Black Measles)"
        elif "Isariopsis" in disease_raw:
            disease = "Leaf Blight (Isariopsis Leaf Spot)"
        else:
            disease = disease_raw.replace("_", " ").strip().title()

    return crop, disease, is_healthy


def initialize_ai_model():
    """Attempt to load TensorFlow model and class labels.

    If either is not available, enable DEMO_MODE.
    """
    global AI_MODEL, CLASS_LABELS, DEMO_MODE, DISEASES_DATA

    DISEASES_DATA = load_disease_database()

    # Potential model paths (check root models/ and backend/models/)
    potential_model_paths = [
        os.path.join(PROJECT_ROOT, "models", "crop_disease_model.h5"),
        os.path.join(BACKEND_DIR, "models", "crop_disease_model.h5"),
    ]
    potential_labels_paths = [
        os.path.join(PROJECT_ROOT, "models", "class_labels.json"),
        os.path.join(BACKEND_DIR, "models", "class_labels.json"),
    ]

    model_path = next((p for p in potential_model_paths if os.path.exists(p)), None)
    labels_path = next((p for p in potential_labels_paths if os.path.exists(p)), None)

    # Load class labels
    if labels_path:
        try:
            with open(labels_path, "r", encoding="utf-8") as f:
                CLASS_LABELS = json.load(f)
            logger.info("Loaded %d class labels from %s", len(CLASS_LABELS), labels_path)
        except Exception as e:
            logger.warning("Could not read class labels file %s: %s", labels_path, str(e))
            CLASS_LABELS = list(DISEASES_DATA.keys())
    else:
        CLASS_LABELS = list(DISEASES_DATA.keys())
        logger.warning("class_labels.json not found; falling back to %d classes from diseases.json", len(CLASS_LABELS))

    # Try loading TensorFlow model
    if model_path:
        try:
            import tensorflow as tf

            logger.info("Loading TensorFlow model from: %s", model_path)
            AI_MODEL = tf.keras.models.load_model(model_path)
            DEMO_MODE = False
            logger.info("TensorFlow model loaded successfully.")
        except Exception as e:
            logger.warning("Failed to load TensorFlow model (%s). Running in DEMO MODE: %s", model_path, str(e))
            AI_MODEL = None
            DEMO_MODE = True
    else:
        logger.warning("Model file 'crop_disease_model.h5' not found. CropGuard AI is starting in DEMO MODE.")
        AI_MODEL = None
        DEMO_MODE = True


def generate_demo_predictions() -> List[Dict[str, Any]]:
    """Generate realistic simulated predictions when in demo mode."""
    available_classes = CLASS_LABELS if CLASS_LABELS else list(DISEASES_DATA.keys())
    if not available_classes:
        available_classes = ["Tomato___Late_blight", "Tomato___Early_blight", "Tomato___healthy"]

    # Pick 3 random distinct classes
    sample_size = min(3, len(available_classes))
    selected_classes = random.sample(available_classes, sample_size)

    # Assign plausible confidence distributions
    conf1 = round(random.uniform(0.85, 0.97), 4)
    conf2 = round(random.uniform(0.02, min(0.12, 1.0 - conf1)), 4)
    conf3 = round(max(0.001, 1.0 - conf1 - conf2), 4)
    confidences = [conf1, conf2, conf3]

    results = []
    for cls_name, conf in zip(selected_classes, confidences):
        crop, disease, is_healthy = parse_class_name(cls_name)
        results.append({
            "class_name": cls_name,
            "crop": crop,
            "disease": disease,
            "confidence": conf,
            "is_healthy": is_healthy,
        })

    # Ensure sorted by confidence DESC
    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results


def create_app(test_config: Optional[Dict[str, Any]] = None) -> Flask:
    """Application factory function for CropGuard AI."""
    app = Flask(
        __name__,
        static_folder=FRONTEND_DIR,
        static_url_path="/static",
    )

    # Enable CORS
    CORS(app)

    # App configuration
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB limit
    app.config["UPLOAD_FOLDER"] = UPLOADS_DIR

    if test_config:
        app.config.update(test_config)

    # Ensure directories exist
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    os.makedirs(os.path.join(BACKEND_DIR, "database"), exist_ok=True)

    # Initialize SQLite database
    init_db()

    # Load model and class info
    initialize_ai_model()

    # -------------------------------------------------------------
    # Global Error Handlers
    # -------------------------------------------------------------
    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({"success": False, "error": str(error.description or "Bad request")}), 400

    @app.errorhandler(404)
    def not_found(error):
        if request.path.startswith("/api/"):
            return jsonify({"success": False, "error": "API endpoint not found"}), 404
        # Serve frontend index for non-API 404s (SPA fallback)
        if os.path.exists(os.path.join(FRONTEND_DIR, "index.html")):
            return send_from_directory(FRONTEND_DIR, "index.html")
        return jsonify({"success": False, "error": "Page not found"}), 404

    @app.errorhandler(413)
    @app.errorhandler(RequestEntityTooLarge)
    def file_too_large(error):
        return jsonify({"success": False, "error": "File size exceeds the 10MB upload limit."}), 413

    @app.errorhandler(500)
    def internal_error(error):
        logger.error("Internal server error: %s", str(error))
        return jsonify({"success": False, "error": "An internal server error occurred."}), 500

    # -------------------------------------------------------------
    # Static & Frontend Routes
    # -------------------------------------------------------------
    @app.route("/", methods=["GET"])
    def index():
        """Serve the frontend single-page application."""
        if os.path.exists(os.path.join(FRONTEND_DIR, "index.html")):
            return send_from_directory(FRONTEND_DIR, "index.html")
        return (
            "<h1>CropGuard AI API is running.</h1>"
            "<p>Place frontend assets in <code>crop-disease-detection/frontend</code> to serve the UI.</p>"
            "<p>API documentation & health: <a href='/api/health'>/api/health</a></p>",
            200,
            {"Content-Type": "text/html"},
        )

    @app.route("/<path:filename>", methods=["GET"])
    def serve_frontend_static(filename):
        """Serve additional frontend files if present (css, js, images)."""
        file_path = os.path.join(FRONTEND_DIR, filename)
        if os.path.exists(file_path) and not os.path.isdir(file_path):
            return send_from_directory(FRONTEND_DIR, filename)
        # If it's index or unknown route, fallback to index.html
        if os.path.exists(os.path.join(FRONTEND_DIR, "index.html")):
            return send_from_directory(FRONTEND_DIR, "index.html")
        return jsonify({"success": False, "error": "Asset not found"}), 404

    @app.route("/uploads/<path:filename>", methods=["GET"])
    def serve_uploaded_file(filename):
        """Serve uploaded leaf image files."""
        return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

    # -------------------------------------------------------------
    # API Routes
    # -------------------------------------------------------------
    @app.route("/api/health", methods=["GET"])
    def health_check():
        """Health check endpoint returning system status and model readiness."""
        total_classes = len(CLASS_LABELS) if CLASS_LABELS else len(DISEASES_DATA)
        return jsonify({
            "status": "ok",
            "model_loaded": AI_MODEL is not None,
            "demo_mode": DEMO_MODE,
            "total_classes": total_classes,
            "version": "1.0.0",
        }), 200

    @app.route("/api/predict", methods=["POST"])
    def predict():
        """Accept image file upload, run inference or demo prediction, and save history."""
        # 1. Check if file is included
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file field found in request."}), 400

        file = request.files["file"]
        if not file or not file.filename or file.filename.strip() == "":
            return jsonify({"success": False, "error": "No file selected for upload."}), 400

        original_filename = file.filename
        if not is_allowed_file(original_filename):
            return jsonify({
                "success": False,
                "error": "Invalid file type. Allowed formats: jpg, jpeg, png, webp.",
            }), 400

        # 2. Secure and save file with unique UUID
        safe_name = secure_filename(original_filename)
        unique_filename = f"{uuid.uuid4().hex}_{safe_name}"
        saved_file_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)

        try:
            file.save(saved_file_path)
        except Exception as e:
            logger.error("Failed to save uploaded file: %s", str(e))
            return jsonify({"success": False, "error": "Could not save uploaded file."}), 500

        # 3. Validate image dimensions and content
        is_valid, validation_msg = validate_image(saved_file_path)
        if not is_valid:
            try:
                os.remove(saved_file_path)
            except OSError:
                pass
            return jsonify({"success": False, "error": validation_msg}), 400

        # 4. Inference or Demo simulation
        top_predictions: List[Dict[str, Any]] = []

        if not DEMO_MODE and AI_MODEL is not None:
            try:
                # Preprocess image for 224x224 RGB normalized input
                processed_img = preprocess_image(saved_file_path, target_size=(224, 224))
                raw_predictions = AI_MODEL.predict(processed_img)[0]

                # Pair with class labels
                indexed_preds = []
                for idx, conf in enumerate(raw_predictions):
                    cls_name = CLASS_LABELS[idx] if idx < len(CLASS_LABELS) else f"Class_{idx}"
                    crop, disease, is_healthy = parse_class_name(cls_name)
                    indexed_preds.append({
                        "class_name": cls_name,
                        "crop": crop,
                        "disease": disease,
                        "confidence": float(round(conf, 4)),
                        "is_healthy": is_healthy,
                    })

                # Sort descending by confidence and take top 3
                indexed_preds.sort(key=lambda x: x["confidence"], reverse=True)
                top_predictions = indexed_preds[:3]
            except Exception as e:
                logger.error("Model prediction failed: %s. Falling back to demo mode for this request.", str(e))
                top_predictions = generate_demo_predictions()
        else:
            top_predictions = generate_demo_predictions()

        top_result = top_predictions[0]
        class_name = top_result["class_name"]
        crop = top_result["crop"]
        disease = top_result["disease"]
        confidence = top_result["confidence"]
        is_healthy = top_result["is_healthy"]

        # 5. Look up comprehensive disease info
        disease_info = DISEASES_DATA.get(class_name)
        if not disease_info:
            disease_info = {
                "class_name": class_name,
                "crop": crop,
                "disease": disease,
                "scientific_name": "N/A",
                "is_healthy": is_healthy,
                "description": f"Details for {disease} on {crop}.",
                "symptoms": ["Consult local agricultural advisor for full analysis."],
                "causes": ["Pathogen infection or cultural condition."],
                "prevention": ["Maintain balanced nutrition and pest control."],
                "treatment": ["Apply targeted remedies based on diagnosis."],
                "management": ["Practice regular crop scouting and field sanitation."],
            }

        # 6. Save prediction to SQLite database
        now_iso = datetime.datetime.now().isoformat()
        db_payload = {
            "image_filename": unique_filename,
            "original_filename": safe_name,
            "predicted_class": class_name,
            "crop_name": crop,
            "disease_name": disease,
            "confidence": confidence,
            "is_healthy": is_healthy,
            "top_predictions": top_predictions,
            "demo_mode": DEMO_MODE or (AI_MODEL is None),
        }

        try:
            record_id = save_prediction(db_payload)
        except Exception as e:
            logger.error("Failed to persist prediction record to database: %s", str(e))
            record_id = None

        # 7. Formulate response
        response_payload = {
            "success": True,
            "prediction": {
                "id": record_id,
                "class_name": class_name,
                "crop": crop,
                "disease": disease,
                "confidence": confidence,
                "is_healthy": is_healthy,
                "top_predictions": top_predictions,
                "image_filename": unique_filename,
                "disease_info": disease_info,
            },
            "disease_info": disease_info,
            "demo_mode": DEMO_MODE or (AI_MODEL is None),
            "timestamp": now_iso,
        }

        return jsonify(response_payload), 200

    @app.route("/api/history", methods=["GET"])
    def get_history():
        """Retrieve all prediction records ordered chronologically descending."""
        try:
            records = get_all_predictions()
            return jsonify({
                "success": True,
                "history": records,
                "total": len(records),
            }), 200
        except Exception as e:
            logger.error("Error retrieving history: %s", str(e))
            return jsonify({"success": False, "error": "Failed to fetch prediction history."}), 500

    @app.route("/api/history/<int:pred_id>", methods=["DELETE"])
    def delete_single_history(pred_id: int):
        """Delete a single prediction record by ID."""
        try:
            existing = get_prediction_by_id(pred_id)
            if not existing:
                return jsonify({"success": False, "error": f"Prediction record with ID {pred_id} not found."}), 404

            # Attempt to delete stored image file if exists
            img_file = existing.get("image_filename")
            if img_file:
                img_path = os.path.join(app.config["UPLOAD_FOLDER"], img_file)
                if os.path.exists(img_path):
                    try:
                        os.remove(img_path)
                    except OSError:
                        pass

            deleted = delete_prediction(pred_id)
            if deleted:
                return jsonify({"success": True, "message": "Record deleted"}), 200
            return jsonify({"success": False, "error": "Could not delete record."}), 500
        except Exception as e:
            logger.error("Error deleting history record %s: %s", pred_id, str(e))
            return jsonify({"success": False, "error": "Internal error deleting record."}), 500

    @app.route("/api/history", methods=["DELETE"])
    def clear_all_history():
        """Delete all prediction records and optionally cleanup uploaded images."""
        try:
            # Delete uploaded files
            try:
                for fname in os.listdir(app.config["UPLOAD_FOLDER"]):
                    fpath = os.path.join(app.config["UPLOAD_FOLDER"], fname)
                    if os.path.isfile(fpath):
                        os.remove(fpath)
            except Exception as file_err:
                logger.warning("Error cleaning up upload directory: %s", str(file_err))

            deleted_count = delete_all_predictions()
            return jsonify({
                "success": True,
                "message": "All history cleared",
                "deleted_count": deleted_count,
            }), 200
        except Exception as e:
            logger.error("Error clearing all history: %s", str(e))
            return jsonify({"success": False, "error": "Failed to clear history."}), 500

    @app.route("/api/diseases", methods=["GET"])
    def get_diseases():
        """Return diseases library with optional filtering by crop or search term."""
        crop_filter = request.args.get("crop", "").strip()
        search_filter = request.args.get("search", "").strip().lower()

        # Extract unique crop list
        unique_crops = sorted({entry.get("crop") for entry in DISEASES_DATA.values() if entry.get("crop")})

        filtered: Dict[str, Any] = {}
        for key, entry in DISEASES_DATA.items():
            # Crop match filter
            if crop_filter:
                entry_crop = entry.get("crop", "")
                if crop_filter.lower() not in entry_crop.lower():
                    continue

            # Search text filter
            if search_filter:
                searchable_text = " ".join([
                    entry.get("class_name", ""),
                    entry.get("crop", ""),
                    entry.get("disease", ""),
                    entry.get("scientific_name", ""),
                    entry.get("description", ""),
                    " ".join(entry.get("symptoms", [])),
                    " ".join(entry.get("causes", [])),
                    " ".join(entry.get("prevention", [])),
                    " ".join(entry.get("treatment", [])),
                ]).lower()

                if search_filter not in searchable_text:
                    continue

            filtered[key] = entry

        return jsonify({
            "success": True,
            "diseases": filtered,
            "crops": unique_crops,
            "total": len(filtered),
        }), 200

    @app.route("/api/dashboard", methods=["GET"])
    def get_dashboard():
        """Retrieve aggregated dashboard metrics, recent predictions, and top diseases."""
        try:
            stats = get_dashboard_stats()
            return jsonify({"success": True, "stats": stats}), 200
        except Exception as e:
            logger.error("Error generating dashboard stats: %s", str(e))
            return jsonify({"success": False, "error": "Failed to retrieve dashboard statistics."}), 500

    return app


# Create default application instance
app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
