import json
from pathlib import Path

import numpy as np
import pandas as pd


TRAIN_PATH = Path(
    "data/processed/fraud_baseline.csv"
)

OUTPUT_PATH = Path(
    "data/processed/psi_reference.json"
)

DRIFT_FEATURES = [
    "TransactionAmt",
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
    "dist1",
]

BINS = 10


def main():

    print("Loading reference dataset...")
    print(f"Source: {TRAIN_PATH}")

    df = pd.read_csv(
        TRAIN_PATH,
        usecols=DRIFT_FEATURES
    )

    reference = {}

    for feature in DRIFT_FEATURES:

        values = pd.to_numeric(
            df[feature],
            errors="coerce"
        ).dropna().values

        # Same percentile logic used by
        # the original drift_monitoring.py
        breakpoints = np.percentile(
            values,
            np.linspace(
                0,
                100,
                BINS + 1
            )
        )

        breakpoints = np.unique(
            breakpoints
        )

        # Same histogram logic used by
        # the original PSI implementation
        if len(breakpoints) >= 3:

            reference_counts, _ = np.histogram(
                values,
                bins=breakpoints
            )

            reference_pct = np.clip(
                reference_counts / len(values),
                0.0001,
                None
            )

        else:

            reference_counts = np.array(
                [],
                dtype=int
            )

            reference_pct = np.array(
                [],
                dtype=float
            )

        reference[feature] = {
            "count": int(len(values)),
            "breakpoints": [
                float(value)
                for value in breakpoints
            ],
            "reference_counts": [
                int(value)
                for value in reference_counts
            ],
            "reference_pct": [
                float(value)
                for value in reference_pct
            ],
        }

        print(
            f"{feature:<20} "
            f"reference_count={len(values)} "
            f"breakpoints={len(breakpoints)} "
            f"histogram_bins={len(reference_counts)}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {
                "version": 2,
                "bins": BINS,
                "features": reference,
            },
            file,
            indent=2,
        )

    print()
    print(
        "PSI reference created successfully."
    )
    print(
        f"Output: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()