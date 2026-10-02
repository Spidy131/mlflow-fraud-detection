import os
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path


MONITORING_DB = Path("data/monitoring.db")
STATE_FILE = Path("data/monitoring_state.txt")

SCRIPTS = [
    ("Data Quality Monitoring", "data_quality_monitoring.py"),
    ("Drift Monitoring", "drift_monitoring.py"),
    ("Alert Monitoring", "monitoring_alerts.py"),
]


def get_current_max_prediction_id():
    """Get the latest prediction ID from the monitoring database."""

    if not MONITORING_DB.exists():
        raise FileNotFoundError(
            f"Monitoring database not found: {MONITORING_DB}"
        )

    conn = sqlite3.connect(MONITORING_DB)

    row = conn.execute(
        "SELECT MAX(id) FROM predictions"
    ).fetchone()

    conn.close()

    return row[0] or 0


def get_last_processed_id():
    """Read the last successfully processed prediction ID."""

    if not STATE_FILE.exists():
        return 0

    try:
        value = STATE_FILE.read_text(
            encoding="utf-8"
        ).strip()

        return int(value)

    except (ValueError, OSError):
        return 0


def save_last_processed_id(prediction_id):
    """Save the latest successfully processed prediction ID."""

    STATE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    STATE_FILE.write_text(
        str(prediction_id),
        encoding="utf-8"
    )


def run_script(
    name,
    script,
    start_id,
    end_id
):
    """Run one monitoring script for the current prediction window."""

    print()
    print("=" * 70)
    print(f"STARTING: {name}")
    print(f"SCRIPT:   {script}")
    print(f"ID RANGE: {start_id + 1} - {end_id}")
    print("=" * 70)

    env = dict(os.environ)

    # Force UTF-8 for child processes on Windows.
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    # Tell monitoring scripts which prediction window to process.
    env["MONITORING_START_ID"] = str(start_id)
    env["MONITORING_END_ID"] = str(end_id)

    result = subprocess.run(
        [sys.executable, script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    if result.stdout:
        print(result.stdout)

    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        print()
        print(f"FAILED: {name}")
        print(f"Exit code: {result.returncode}")
        return False

    print()
    print(f"COMPLETED: {name}")

    return True


def main():

    start_time = datetime.now()

    print("=" * 70)
    print("FRAUD DETECTION - INCREMENTAL MONITORING PIPELINE")
    print("=" * 70)
    print(f"Started: {start_time}")

    current_max_id = get_current_max_prediction_id()
    last_processed_id = get_last_processed_id()

    print()
    print(f"Last processed prediction ID: {last_processed_id}")
    print(f"Current maximum prediction ID: {current_max_id}")

    # --------------------------------------------------------
    # No new predictions
    # --------------------------------------------------------

    if current_max_id <= last_processed_id:

        print()
        print("=" * 70)
        print("NO NEW PREDICTIONS TO MONITOR")
        print("=" * 70)
        print("Pipeline skipped.")

        return

    new_start_id = last_processed_id
    new_end_id = current_max_id

    new_prediction_count = (
        current_max_id - last_processed_id
    )

    print()
    print(f"New predictions to monitor: {new_prediction_count}")
    print(
        f"Monitoring prediction IDs: "
        f"{new_start_id + 1} - {new_end_id}"
    )

    # --------------------------------------------------------
    # Run monitoring stages
    # --------------------------------------------------------

    for name, script in SCRIPTS:

        success = run_script(
            name,
            script,
            new_start_id,
            new_end_id,
        )

        if not success:

            print()
            print("=" * 70)
            print("MONITORING PIPELINE FAILED")
            print("=" * 70)

            # IMPORTANT:
            # Do not update the state file when a stage fails.
            # The same window can be retried on the next run.

            sys.exit(1)

    # --------------------------------------------------------
    # All stages succeeded
    # --------------------------------------------------------

    save_last_processed_id(
        current_max_id
    )

    end_time = datetime.now()
    duration = end_time - start_time

    print()
    print("=" * 70)
    print("MONITORING PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"Processed IDs: {new_start_id + 1} - {new_end_id}")
    print(f"Processed rows: {new_prediction_count}")
    print(f"Started:        {start_time}")
    print(f"Finished:       {end_time}")
    print(f"Duration:       {duration}")
    print(
        f"State saved:    {STATE_FILE}"
    )


if __name__ == "__main__":
    main()