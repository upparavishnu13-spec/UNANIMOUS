"""Database module for CropGuard AI.

Handles SQLite connection, schema creation, and CRUD operations for predictions.
"""

import json
import logging
import os
import sqlite3
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default database file path: backend/database/cropguard.db
DEFAULT_DB_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "cropguard.db")
)


def get_db_path(custom_path: Optional[str] = None) -> str:
    """Return the absolute path to the database file."""
    if custom_path:
        return os.path.abspath(custom_path)
    return DEFAULT_DB_PATH


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Create and return a database connection with Row factory enabled."""
    target_path = get_db_path(db_path)
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    conn = sqlite3.connect(target_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    """Initialize the database schema if tables do not already exist."""
    schema = """
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_filename TEXT NOT NULL,
        original_filename TEXT NOT NULL,
        predicted_class TEXT NOT NULL,
        crop_name TEXT NOT NULL,
        disease_name TEXT NOT NULL,
        confidence REAL NOT NULL,
        is_healthy BOOLEAN NOT NULL DEFAULT 0,
        top_predictions TEXT,
        demo_mode BOOLEAN DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    try:
        with get_connection(db_path) as conn:
            conn.execute(schema)
            conn.commit()
            logger.info("Database initialized successfully at: %s", get_db_path(db_path))
    except Exception as e:
        logger.error("Failed to initialize database: %s", str(e))
        raise


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    """Convert a sqlite3.Row object to a clean dictionary with parsed JSON fields."""
    item = dict(row)

    # Convert SQLite boolean integer representations to Python booleans
    item["is_healthy"] = bool(item.get("is_healthy", 0))
    item["demo_mode"] = bool(item.get("demo_mode", 0))

    # Parse JSON string for top_predictions
    top_preds_raw = item.get("top_predictions")
    if top_preds_raw and isinstance(top_preds_raw, str):
        try:
            item["top_predictions"] = json.loads(top_preds_raw)
        except (json.JSONDecodeError, TypeError):
            item["top_predictions"] = []
    elif not top_preds_raw:
        item["top_predictions"] = []

    return item


def save_prediction(data_dict: Dict[str, Any], db_path: Optional[str] = None) -> int:
    """Save a prediction record into the database.

    Args:
        data_dict: Dictionary containing prediction details.
        db_path: Optional custom path to SQLite database.

    Returns:
        The inserted row ID.
    """
    query = """
    INSERT INTO predictions (
        image_filename,
        original_filename,
        predicted_class,
        crop_name,
        disease_name,
        confidence,
        is_healthy,
        top_predictions,
        demo_mode
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    top_predictions = data_dict.get("top_predictions", [])
    if not isinstance(top_predictions, str):
        top_predictions_json = json.dumps(top_predictions)
    else:
        top_predictions_json = top_predictions

    params = (
        data_dict.get("image_filename", ""),
        data_dict.get("original_filename", ""),
        data_dict.get("predicted_class", ""),
        data_dict.get("crop_name", ""),
        data_dict.get("disease_name", ""),
        float(data_dict.get("confidence", 0.0)),
        1 if data_dict.get("is_healthy") else 0,
        top_predictions_json,
        1 if data_dict.get("demo_mode") else 0,
    )

    try:
        with get_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            inserted_id = cursor.lastrowid
            logger.info("Saved prediction id=%s for class=%s", inserted_id, data_dict.get("predicted_class"))
            return inserted_id
    except Exception as e:
        logger.error("Error saving prediction: %s", str(e))
        raise


def get_all_predictions(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetch all predictions ordered by created_at DESC."""
    query = "SELECT * FROM predictions ORDER BY created_at DESC, id DESC"
    try:
        with get_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            return [_row_to_dict(row) for row in rows]
    except Exception as e:
        logger.error("Error fetching all predictions: %s", str(e))
        return []


def get_prediction_by_id(pred_id: int, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Fetch a single prediction record by ID."""
    query = "SELECT * FROM predictions WHERE id = ?"
    try:
        with get_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, (pred_id,))
            row = cursor.fetchone()
            if row:
                return _row_to_dict(row)
            return None
    except Exception as e:
        logger.error("Error fetching prediction id %s: %s", pred_id, str(e))
        return None


def delete_prediction(pred_id: int, db_path: Optional[str] = None) -> bool:
    """Delete a specific prediction record by ID.

    Returns:
        True if a record was deleted, False otherwise.
    """
    query = "DELETE FROM predictions WHERE id = ?"
    try:
        with get_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, (pred_id,))
            conn.commit()
            return cursor.rowcount > 0
    except Exception as e:
        logger.error("Error deleting prediction id %s: %s", pred_id, str(e))
        return False


def delete_all_predictions(db_path: Optional[str] = None) -> int:
    """Delete all prediction records from the database.

    Returns:
        The number of records deleted.
    """
    query = "DELETE FROM predictions"
    try:
        with get_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            conn.commit()
            deleted_count = cursor.rowcount
            logger.info("Deleted %d prediction records", deleted_count)
            return deleted_count
    except Exception as e:
        logger.error("Error deleting all predictions: %s", str(e))
        return 0


def get_dashboard_stats(db_path: Optional[str] = None) -> Dict[str, Any]:
    """Calculate and return dashboard statistics.

    Returns:
        dict with:
            - total_predictions: total count
            - healthy_count: count of healthy predictions
            - diseased_count: count of diseased predictions
            - demo_count: count of demo-mode predictions
            - top_diseases: list of {"disease": name, "count": count} sorted DESC
            - recent: list of the 10 most recent predictions
    """
    stats = {
        "total_predictions": 0,
        "healthy_count": 0,
        "diseased_count": 0,
        "demo_count": 0,
        "top_diseases": [],
        "recent": [],
    }

    try:
        with get_connection(db_path) as conn:
            cursor = conn.cursor()

            # Aggregate counts
            cursor.execute(
                """
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN is_healthy = 1 THEN 1 ELSE 0 END) as healthy,
                    SUM(CASE WHEN is_healthy = 0 THEN 1 ELSE 0 END) as diseased,
                    SUM(CASE WHEN demo_mode = 1 THEN 1 ELSE 0 END) as demo
                FROM predictions
                """
            )
            agg = cursor.fetchone()
            if agg:
                stats["total_predictions"] = agg["total"] or 0
                stats["healthy_count"] = agg["healthy"] or 0
                stats["diseased_count"] = agg["diseased"] or 0
                stats["demo_count"] = agg["demo"] or 0

            # Top diseases (excluding Healthy entries for meaningful disease charts, but fallback to all if needed)
            cursor.execute(
                """
                SELECT disease_name as disease, COUNT(*) as count
                FROM predictions
                WHERE is_healthy = 0 AND disease_name != 'Healthy'
                GROUP BY disease_name
                ORDER BY count DESC
                LIMIT 10
                """
            )
            top_rows = cursor.fetchall()
            stats["top_diseases"] = [
                {"disease": row["disease"], "count": row["count"]} for row in top_rows
            ]

            # Recent 10 predictions
            cursor.execute(
                "SELECT * FROM predictions ORDER BY created_at DESC, id DESC LIMIT 10"
            )
            recent_rows = cursor.fetchall()
            stats["recent"] = [_row_to_dict(row) for row in recent_rows]

    except Exception as e:
        logger.error("Error retrieving dashboard stats: %s", str(e))

    return stats
