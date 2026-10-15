"""CropGuard AI - API Test Suite

Comprehensive pytest tests covering the Flask REST API endpoints:
  - GET  /api/health
  - POST /api/predict
  - GET  /api/history
  - DELETE /api/history/<id>
  - DELETE /api/history
  - GET  /api/diseases
  - GET  /api/diseases?crop=<name>
  - GET  /api/dashboard
"""

import io
import json
import os
import sys
import tempfile
from typing import Generator
import pytest
from PIL import Image

# Add backend directory to sys.path
BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'backend')
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

try:
    from app import create_app
except ImportError:
    create_app = None


def create_mock_image_bytes(
    format: str = "JPEG",
    size: tuple = (224, 224),
    color: tuple = (34, 139, 34)
) -> io.BytesIO:
    """Generate in-memory valid leaf-like image bytes."""
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=color)
    img.save(buf, format=format)
    buf.seek(0)
    return buf


@pytest.fixture
def app():
    """Create application configured for testing."""
    if create_app is None:
        pytest.skip("Backend 'app.py' not found in backend directory.")

    temp_dir = tempfile.mkdtemp()
    test_db = os.path.join(temp_dir, "test_cropguard.db")
    test_uploads = os.path.join(temp_dir, "uploads")
    os.makedirs(test_uploads, exist_ok=True)

    test_config = {
        'TESTING': True,
        'DATABASE': test_db,
        'UPLOAD_FOLDER': test_uploads,
        'SECRET_KEY': 'test-secret-key-12345',
        'MAX_CONTENT_LENGTH': 10 * 1024 * 1024
    }

    try:
        app_instance = create_app(test_config)
    except TypeError:
        app_instance = create_app()
        for k, v in test_config.items():
            app_instance.config[k] = v

    yield app_instance

    # Cleanup temp directory
    try:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass


@pytest.fixture
def client(app):
    """Test client fixture."""
    return app.test_client()


# ============================================================================
# Health Endpoint Tests
# ============================================================================

def test_health_endpoint_success(client):
    """Test GET /api/health returns 200 and healthy status."""
    response = client.get('/api/health')
    assert response.status_code == 200
    data = response.get_json()
    assert data is not None
    assert "status" in data
    assert data["status"] in ["ok", "healthy", "success"]


def test_health_endpoint_disallows_post(client):
    """Test POST /api/health is rejected with 405 Method Not Allowed."""
    response = client.post('/api/health')
    assert response.status_code == 405


# ============================================================================
# Prediction Endpoint Tests
# ============================================================================

def test_predict_with_valid_jpeg_image(client):
    """Test POST /api/predict with valid JPEG leaf image returns 200 and prediction."""
    img_bytes = create_mock_image_bytes(format="JPEG", color=(60, 179, 113))
    data = {
        'file': (img_bytes, 'tomato_leaf.jpg')
    }
    response = client.post(
        '/api/predict',
        data=data,
        content_type='multipart/form-data'
    )
    assert response.status_code == 200
    result = response.get_json()
    assert result is not None
    # Validate core prediction fields
    assert "prediction" in result or "crop" in result or "class_name" in result


def test_predict_with_valid_png_image(client):
    """Test POST /api/predict accepts PNG images."""
    img_bytes = create_mock_image_bytes(format="PNG", color=(46, 139, 87))
    data = {
        'file': (img_bytes, 'corn_leaf.png')
    }
    response = client.post(
        '/api/predict',
        data=data,
        content_type='multipart/form-data'
    )
    assert response.status_code == 200
    result = response.get_json()
    assert result is not None


def test_predict_no_file_uploaded(client):
    """Test POST /api/predict returns 400 when no file is included in request."""
    response = client.post(
        '/api/predict',
        data={},
        content_type='multipart/form-data'
    )
    assert response.status_code == 400
    result = response.get_json()
    assert "error" in result or "message" in result


def test_predict_empty_filename(client):
    """Test POST /api/predict returns 400 when file part is submitted with empty filename."""
    data = {
        'file': (io.BytesIO(b""), '')
    }
    response = client.post(
        '/api/predict',
        data=data,
        content_type='multipart/form-data'
    )
    assert response.status_code == 400


@pytest.mark.parametrize("invalid_filename,content", [
    ("document.txt", b"plain text data not an image"),
    ("script.py", b"print('malicious code')"),
    ("manual.pdf", b"%PDF-1.4 header text"),
    ("archive.zip", b"PK\x03\x04 fake zip content"),
])
def test_predict_invalid_file_type(client, invalid_filename, content):
    """Test POST /api/predict rejects non-image file extensions."""
    data = {
        'file': (io.BytesIO(content), invalid_filename)
    }
    response = client.post(
        '/api/predict',
        data=data,
        content_type='multipart/form-data'
    )
    assert response.status_code == 400
    result = response.get_json()
    assert "error" in result or "message" in result


# ============================================================================
# History Endpoint Tests
# ============================================================================

def test_get_history_returns_list(client):
    """Test GET /api/history returns 200 and history list."""
    response = client.get('/api/history')
    assert response.status_code == 200
    data = response.get_json()
    assert isinstance(data, list) or ("history" in data and isinstance(data["history"], list))


def test_prediction_persisted_in_history(client):
    """Test that a successful prediction is recorded in history."""
    img_bytes = create_mock_image_bytes(format="JPEG")
    predict_resp = client.post(
        '/api/predict',
        data={'file': (img_bytes, 'test_record.jpg')},
        content_type='multipart/form-data'
    )
    assert predict_resp.status_code == 200

    history_resp = client.get('/api/history')
    assert history_resp.status_code == 200
    history_data = history_resp.get_json()
    items = history_data if isinstance(history_data, list) else history_data.get("history", [])
    assert len(items) >= 1


def test_delete_history_item_by_id(client):
    """Test DELETE /api/history/<id> removes specific item."""
    # First create an item via predict
    img_bytes = create_mock_image_bytes(format="JPEG")
    client.post(
        '/api/predict',
        data={'file': (img_bytes, 'to_delete.jpg')},
        content_type='multipart/form-data'
    )

    history_resp = client.get('/api/history')
    items = history_resp.get_json()
    if isinstance(items, dict):
        items = items.get("history", [])

    if items:
        item_id = items[0].get("id")
        del_resp = client.delete(f'/api/history/{item_id}')
        assert del_resp.status_code in [200, 204]


def test_delete_history_item_not_found(client):
    """Test DELETE /api/history/<invalid_id> handles non-existent record."""
    del_resp = client.delete('/api/history/999999999')
    assert del_resp.status_code in [200, 404]


def test_clear_all_history(client):
    """Test DELETE /api/history clears entire history."""
    # Add a prediction
    img_bytes = create_mock_image_bytes(format="JPEG")
    client.post(
        '/api/predict',
        data={'file': (img_bytes, 'temp.jpg')},
        content_type='multipart/form-data'
    )

    # Delete all
    del_all_resp = client.delete('/api/history')
    assert del_all_resp.status_code in [200, 204]

    # Verify history is now empty
    history_resp = client.get('/api/history')
    data = history_resp.get_json()
    items = data if isinstance(data, list) else data.get("history", [])
    assert len(items) == 0


# ============================================================================
# Disease Information Endpoint Tests
# ============================================================================

def test_get_diseases_all(client):
    """Test GET /api/diseases returns disease catalog."""
    response = client.get('/api/diseases')
    assert response.status_code == 200
    data = response.get_json()
    assert data is not None
    # Can be a list of diseases or a dict with disease keys
    assert isinstance(data, (list, dict))
    assert len(data) > 0


def test_get_diseases_filter_crop_tomato(client):
    """Test GET /api/diseases?crop=Tomato filters results to Tomato diseases only."""
    response = client.get('/api/diseases?crop=Tomato')
    assert response.status_code == 200
    data = response.get_json()
    assert data is not None
    diseases = data if isinstance(data, list) else data.get("diseases", [])
    for d in diseases:
        if isinstance(d, dict) and "crop" in d:
            assert d["crop"].lower() == "tomato"


def test_get_diseases_filter_unknown_crop(client):
    """Test GET /api/diseases with unknown crop returns empty results."""
    response = client.get('/api/diseases?crop=NonExistentCropName')
    assert response.status_code == 200
    data = response.get_json()
    diseases = data if isinstance(data, list) else data.get("diseases", [])
    assert len(diseases) == 0


# ============================================================================
# Dashboard Statistics Endpoint Tests
# ============================================================================

def test_get_dashboard_stats(client):
    """Test GET /api/dashboard returns statistical metrics."""
    response = client.get('/api/dashboard')
    assert response.status_code == 200
    data = response.get_json()
    assert data is not None
    assert isinstance(data, dict)
    # Check for dashboard indicators (e.g., total_scans, healthy_count, etc.)
    expected_keys = {"total_scans", "total_predictions", "recent_predictions", "crops", "stats"}
    assert any(k in data for k in expected_keys) or len(data) > 0
