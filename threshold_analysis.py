import pandas as pd
import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# Configuration
# ============================================================

RESULTS_FILE = "api_holdout_results_2000.csv"

THRESHOLDS = [
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.75,
    0.78,
    0.80,
    0.85,
    0.90,
]


# ============================================================
# Load API results
# ============================================================

df = pd.read_csv(RESULTS_FILE)

print("=" * 80)
print("THRESHOLD ANALYSIS")
print("=" * 80)

print(f"\nLoaded: {RESULTS_FILE}")
print(f"Rows: {len(df)}")

# Expected columns:
# probability
# actual
#
# If your CSV uses different names, change these two lines.

y_true = df["actual"].astype(int)
y_probability = df["probability"].astype(float)


# ============================================================
# Analyze thresholds
# ============================================================

results = []

for threshold in THRESHOLDS:

    y_pred = (y_probability >= threshold).astype(int)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    false_positive_rate = fp / (fp + tn)

    predicted_fraud = int(y_pred.sum())

    results.append({
        "Threshold": threshold,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "False Positives": fp,
        "False Negatives": fn,
        "False Positive Rate": false_positive_rate,
        "Predicted Fraud": predicted_fraud,
        "TP": tp,
        "TN": tn,
    })


results_df = pd.DataFrame(results)


# ============================================================
# Display results
# ============================================================

print("\nThreshold Performance:")
print("-" * 120)

print(
    results_df.to_string(
        index=False,
        formatters={
            "Threshold": "{:.2f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1": "{:.4f}".format,
            "False Positive Rate": "{:.4f}".format,
        }
    )
)


# ============================================================
# Best F1 threshold
# ============================================================

best_f1_row = results_df.loc[
    results_df["F1"].idxmax()
]

print("\n" + "=" * 80)
print("BEST F1 THRESHOLD")
print("=" * 80)

print(
    f"Threshold       : {best_f1_row['Threshold']:.2f}"
)
print(
    f"Precision       : {best_f1_row['Precision']:.4f}"
)
print(
    f"Recall          : {best_f1_row['Recall']:.4f}"
)
print(
    f"F1              : {best_f1_row['F1']:.4f}"
)
print(
    f"False Positives : {int(best_f1_row['False Positives'])}"
)
print(
    f"False Negatives : {int(best_f1_row['False Negatives'])}"
)
print(
    f"False Positive Rate : {best_f1_row['False Positive Rate']:.4f}"
)


# ============================================================
# Best threshold with Precision >= 80%
# ============================================================

precision_target = 0.80

precision_candidates = results_df[
    results_df["Precision"] >= precision_target
]

if not precision_candidates.empty:

    best_precision_row = precision_candidates.loc[
        precision_candidates["Recall"].idxmax()
    ]

    print("\n" + "=" * 80)
    print("BEST THRESHOLD WITH PRECISION >= 80%")
    print("=" * 80)

    print(
        f"Threshold       : {best_precision_row['Threshold']:.2f}"
    )
    print(
        f"Precision       : {best_precision_row['Precision']:.4f}"
    )
    print(
        f"Recall          : {best_precision_row['Recall']:.4f}"
    )
    print(
        f"F1              : {best_precision_row['F1']:.4f}"
    )
    print(
        f"False Positives : {int(best_precision_row['False Positives'])}"
    )
    print(
        f"False Negatives : {int(best_precision_row['False Negatives'])}"
    )


# ============================================================
# Current production threshold
# ============================================================

production_threshold = 0.78

production_row = results_df[
    np.isclose(
        results_df["Threshold"],
        production_threshold
    )
]

if not production_row.empty:

    row = production_row.iloc[0]

    print("\n" + "=" * 80)
    print("CURRENT PRODUCTION THRESHOLD")
    print("=" * 80)

    print(
        f"Threshold       : {production_threshold:.2f}"
    )
    print(
        f"Precision       : {row['Precision']:.4f}"
    )
    print(
        f"Recall          : {row['Recall']:.4f}"
    )
    print(
        f"F1              : {row['F1']:.4f}"
    )
    print(
        f"False Positives : {int(row['False Positives'])}"
    )
    print(
        f"False Negatives : {int(row['False Negatives'])}"
    )
    print(
        f"False Positive Rate : {row['False Positive Rate']:.4f}"
    )
    print(
        f"Predicted Fraud : {int(row['Predicted Fraud'])}"
    )


# ============================================================
# Save analysis
# ============================================================

OUTPUT_FILE = "threshold_analysis_results.csv"

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 80)
print(f"Analysis saved to: {OUTPUT_FILE}")
print("=" * 80)