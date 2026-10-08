import sqlite3
from datetime import datetime, timezone
from pathlib import Path
import json

DB_PATH = Path("/app/data/monitoring.db")


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(
        DB_PATH,
        check_same_thread=False
    )

    conn.row_factory = sqlite3.Row

    return conn


def init_db():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            probability REAL NOT NULL,
            prediction INTEGER NOT NULL,
            threshold REAL NOT NULL,
            result TEXT NOT NULL,
            model_name TEXT NOT NULL,
            model_version TEXT NOT NULL,
            model_alias TEXT NOT NULL,
            model_id TEXT NOT NULL,
            latency_ms REAL NOT NULL,
            input_features TEXT
        )
    """)

    conn.commit()

    # Add input_features column to existing databases
    columns = [
        row["name"]
        for row in cursor.execute(
            "PRAGMA table_info(predictions)"
        ).fetchall()
    ]

    if "input_features" not in columns:

        cursor.execute("""
            ALTER TABLE predictions
            ADD COLUMN input_features TEXT
        """)

        conn.commit()

    conn.close()


def log_prediction(
    probability,
    prediction,
    threshold,
    result,
    model_name,
    model_version,
    model_alias,
    model_id,
    latency_ms,
    input_features=None,
):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO predictions (
            timestamp,
            probability,
            prediction,
            threshold,
            result,
            model_name,
            model_version,
            model_alias,
            model_id,
            latency_ms,
            input_features
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now(timezone.utc).isoformat(),
        probability,
        prediction,
        threshold,
        result,
        model_name,
        str(model_version),
        model_alias,
        model_id,
        latency_ms,
        json.dumps(input_features)
        if input_features is not None
        else None,
    ))

    conn.commit()
    conn.close()


def get_monitoring_summary():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            COUNT(*) AS total,
            SUM(
                CASE
                    WHEN prediction = 1 THEN 1
                    ELSE 0
                END
            ) AS fraud,
            SUM(
                CASE
                    WHEN prediction = 0 THEN 1
                    ELSE 0
                END
            ) AS legitimate,
            AVG(probability) AS average_probability,
            MAX(probability) AS maximum_probability,
            AVG(latency_ms) AS average_latency_ms
        FROM predictions
    """)

    row = cursor.fetchone()

    conn.close()

    total = row["total"] or 0
    fraud = row["fraud"] or 0
    legitimate = row["legitimate"] or 0

    average_probability = (
        row["average_probability"] or 0.0
    )

    maximum_probability = (
        row["maximum_probability"] or 0.0
    )

    average_latency_ms = (
        row["average_latency_ms"] or 0.0
    )

    fraud_rate = (
        fraud / total
        if total > 0
        else 0.0
    )

    return {
        "total": total,
        "fraud": fraud,
        "legitimate": legitimate,
        "fraud_rate": round(fraud_rate, 4),
        "average_probability": round(
            average_probability,
            4
        ),
        "maximum_probability": round(
            maximum_probability,
            4
        ),
        "average_latency_ms": round(
            average_latency_ms,
            2
        ),
    }


def get_recent_predictions(limit=20):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            id,
            timestamp,
            probability,
            prediction,
            threshold,
            result,
            model_name,
            model_version,
            model_alias,
            model_id,
            latency_ms
        FROM predictions
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))

    rows = cursor.fetchall()

    conn.close()

    return [
        dict(row)
        for row in rows
    ]