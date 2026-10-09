import json
import sqlite3

import numpy as np
import pandas as pd


TRAIN_PATH = "data/processed/fraud_baseline.csv"
PSI_REFERENCE_PATH = "data/processed/psi_reference.json"
MONITORING_DB = "data/monitoring.db"

FEATURES = [
    "TransactionAmt",
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
    "dist1",
]


def calculate_original_psi(reference, current):
    reference = pd.to_numeric(
        reference,
        errors="coerce"
    ).dropna().values

    current = pd.to_numeric(
        current,
        errors="coerce"
    ).dropna().values

    if len(reference) == 0 or len(current) == 0:
        return 0.0

    breakpoints = np.percentile(
        reference,
        np.linspace(0, 100, 11)
    )

    breakpoints = np.unique(breakpoints)

    if len(breakpoints) < 3:
        return 0.0

    reference_counts, _ = np.histogram(
        reference,
        bins=breakpoints
    )

    current_counts, _ = np.histogram(
        current,
        bins=breakpoints
    )

    reference_pct = np.clip(
        reference_counts / len(reference),
        0.0001,
        None
    )

    current_pct = np.clip(
        current_counts / len(current),
        0.0001,
        None
    )

    return float(
        np.sum(
            (current_pct - reference_pct)
            * np.log(
                current_pct / reference_pct
            )
        )
    )


def calculate_compact_psi(
    current,
    breakpoints,
    reference_counts
):
    current = pd.to_numeric(
        current,
        errors="coerce"
    ).dropna().values

    breakpoints = np.array(
        breakpoints,
        dtype=float
    )

    reference_counts = np.array(
        reference_counts,
        dtype=float
    )

    if (
        len(current) == 0
        or len(breakpoints) < 3
        or len(reference_counts) == 0
    ):
        return 0.0

    current_counts, _ = np.histogram(
        current,
        bins=breakpoints
    )

    reference_pct = np.clip(
        reference_counts / np.sum(reference_counts),
        0.0001,
        None
    )

    current_pct = np.clip(
        current_counts / len(current),
        0.0001,
        None
    )

    return float(
        np.sum(
            (current_pct - reference_pct)
            * np.log(
                current_pct / reference_pct
            )
        )
    )


print("=" * 70)
print("PSI REFERENCE VALIDATION")
print("=" * 70)

print()
print("Loading original reference dataset...")

reference_df = pd.read_csv(
    TRAIN_PATH,
    usecols=FEATURES
)

print(
    f"Reference rows: {len(reference_df)}"
)

print()
print("Loading compact PSI reference...")

with open(
    PSI_REFERENCE_PATH,
    "r",
    encoding="utf-8"
) as file:
    psi_reference = json.load(file)

print(
    f"Reference version: "
    f"{psi_reference['version']}"
)

print()
print("Loading latest 100 production records...")

conn = sqlite3.connect(MONITORING_DB)

rows = conn.execute(
    """
    SELECT input_features
    FROM predictions
    WHERE input_features IS NOT NULL
    ORDER BY id DESC
    LIMIT 100
    """
).fetchall()

conn.close()

production_records = []

for row in rows:
    production_records.append(
        json.loads(row[0])
    )

production_df = pd.DataFrame(
    production_records
)

print(
    f"Production rows: "
    f"{len(production_df)}"
)

print()
print("Comparing PSI calculations...")
print()

all_match = True

for feature in FEATURES:

    original_psi = calculate_original_psi(
        reference_df[feature],
        production_df[feature]
    )

    compact_feature = (
        psi_reference["features"][feature]
    )

    compact_psi = calculate_compact_psi(
        production_df[feature],
        compact_feature["breakpoints"],
        compact_feature["reference_counts"]
    )

    difference = abs(
        original_psi - compact_psi
    )

    matches = difference < 1e-12

    if not matches:
        all_match = False

    status = "MATCH" if matches else "MISMATCH"

    print(
        f"{feature:<20} "
        f"original={original_psi:.12f} "
        f"compact={compact_psi:.12f} "
        f"difference={difference:.12e} "
        f"{status}"
    )

print()
print("=" * 70)

if all_match:
    print(
        "VALIDATION PASSED"
    )
    print(
        "Compact PSI reference produces "
        "the same PSI values as the original reference dataset."
    )
else:
    print(
        "VALIDATION FAILED"
    )
    print(
        "At least one PSI value differs."
    )

print("=" * 70)