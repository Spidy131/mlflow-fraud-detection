import os
import subprocess
import sys
import time
from datetime import datetime


INTERVAL_SECONDS = int(
    os.getenv("MONITORING_INTERVAL_SECONDS", "3600")
)


def run_monitoring():

    print()
    print("=" * 70)
    print("FRAUD DETECTION - DOCKER MONITORING WORKER")
    print("=" * 70)
    print(f"Started: {datetime.now()}")
    print("Running incremental monitoring pipeline...")

    env = dict(os.environ)

    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    result = subprocess.run(
        [sys.executable, "run_monitoring.py"],
        env=env,
        text=True,
    )

    if result.returncode == 0:
        print()
        print("Monitoring pipeline completed successfully.")
    else:
        print()
        print(
            f"Monitoring pipeline failed "
            f"with exit code {result.returncode}."
        )


def main():

    print("=" * 70)
    print("FRAUD DETECTION - AUTOMATED MONITORING WORKER")
    print("=" * 70)
    print(
        f"Monitoring interval: "
        f"{INTERVAL_SECONDS} seconds"
    )

    while True:

        try:
            run_monitoring()

        except Exception as exc:
            print()
            print("Monitoring worker error:")
            print(exc)

        print()
        print("=" * 70)
        print(
            f"Next monitoring run in "
            f"{INTERVAL_SECONDS} seconds."
        )
        print("=" * 70)

        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    main()