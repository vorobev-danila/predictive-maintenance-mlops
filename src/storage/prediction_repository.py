import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


class PredictionRepository:
    def __init__(self, db_path):
        self.db_path = Path(db_path)

    def initialize(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    input_json TEXT NOT NULL,
                    predicted_rul REAL NOT NULL,
                    actual_rul REAL,
                    anomaly_flag INTEGER NOT NULL DEFAULT 0,
                    model_version TEXT
                )
                """)
            connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_predictions_created_at
                ON predictions(created_at DESC)
                """)

    def create_prediction(
        self,
        input_payload,
        predicted_rul,
        actual_rul=None,
        anomaly_flag=False,
        model_version=None,
    ):
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO predictions (
                    created_at,
                    input_json,
                    predicted_rul,
                    actual_rul,
                    anomaly_flag,
                    model_version
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    json.dumps(input_payload, sort_keys=True),
                    float(predicted_rul),
                    None if actual_rul is None else float(actual_rul),
                    int(bool(anomaly_flag)),
                    model_version,
                ),
            )
            prediction_id = cursor.lastrowid

        return self._get_prediction(prediction_id)

    def list_recent(self, limit=20):
        bounded_limit = max(1, min(int(limit), 100))
        return self._list_recent_bounded(bounded_limit)

    def list_recent_for_drift(self, limit=100):
        bounded_limit = max(1, min(int(limit), 1000))
        return self._list_recent_bounded(bounded_limit)

    def _list_recent_bounded(self, bounded_limit):
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    id,
                    created_at,
                    input_json,
                    predicted_rul,
                    actual_rul,
                    anomaly_flag,
                    model_version
                FROM predictions
                ORDER BY datetime(created_at) DESC, id DESC
                LIMIT ?
                """,
                (bounded_limit,),
            ).fetchall()

        return [self._row_to_dict(row) for row in rows]

    def _get_prediction(self, prediction_id):
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    id,
                    created_at,
                    input_json,
                    predicted_rul,
                    actual_rul,
                    anomaly_flag,
                    model_version
                FROM predictions
                WHERE id = ?
                """,
                (prediction_id,),
            ).fetchone()

        return self._row_to_dict(row)

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _row_to_dict(row):
        if row is None:
            return None

        return {
            "id": row["id"],
            "created_at": row["created_at"],
            "input": json.loads(row["input_json"]),
            "predicted_rul": row["predicted_rul"],
            "actual_rul": row["actual_rul"],
            "anomaly_flag": bool(row["anomaly_flag"]),
            "model_version": row["model_version"],
        }
