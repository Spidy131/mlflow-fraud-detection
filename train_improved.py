import pandas as pd
import numpy as np
import json

from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from xgboost import XGBClassifier
import skops.io as sio


# ============================================================
# Configuration
# ============================================================

DATA_PATH = "data/processed/fraud_baseline.csv"

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5

TARGET = "isFraud"

# ============================================================
# Load dataset
# ============================================================

print("Loading processed fraud dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# ============================================================
# Prepare X / y
# ============================================================

X = df.drop(columns=[TARGET]).copy()
y = df[TARGET].copy()

categorical_features = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

print(f"Numeric features: {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")


# ============================================================
# CREATE AND SAVE CATEGORICAL MAPPINGS
# ============================================================

category_mappings = {}

for column in categorical_features:

    X[column] = (
        X[column]
        .fillna("missing")
        .astype(str)
    )

    # Create stable mapping from entire training dataset
    values = X[column].unique().tolist()

    mapping = {
        value: index
        for index, value in enumerate(values)
    }

    category_mappings[column] = mapping

    X[column] = X[column].map(mapping).astype(int)


# Save mappings
MAPPINGS_PATH = MODEL_DIR / "category_mappings.json"

with open(MAPPINGS_PATH, "w") as f:
    json.dump(category_mappings, f, indent=2)

print(f"Categorical mappings saved: {MAPPINGS_PATH}")


# ============================================================
# Train / Test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE,
)

print(f"Training rows: {len(X_train)}")
print(f"Testing rows : {len(X_test)}")


# ============================================================
# Class imbalance
# ============================================================

negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

scale_pos_weight = negative / positive

print("\nClass distribution")
print("-------------------")
print(f"Legitimate: {negative}")
print(f"Fraud     : {positive}")
print(f"scale_pos_weight: {scale_pos_weight:.4f}")


# ============================================================
# XGBoost configuration
# ============================================================

params = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,

    "subsample": 0.8,
    "colsample_bytree": 0.8,

    # Regularization
    "reg_alpha": 0.1,
    "reg_lambda": 5.0,
    "min_child_weight": 3,

    "objective": "binary:logistic",
    "eval_metric": "aucpr",

    "scale_pos_weight": scale_pos_weight,

    "tree_method": "hist",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


# ============================================================
# Cross Validation
# ============================================================

print("\n========================================")
print("5-FOLD STRATIFIED CROSS VALIDATION")
print("========================================")

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

oof_predictions = np.zeros(len(X_train))

cv_results = []


for fold, (train_idx, val_idx) in enumerate(
    skf.split(X_train, y_train),
    start=1,
):

    print(f"\n========== FOLD {fold} ==========")

    X_fold_train = X_train.iloc[train_idx]
    X_fold_val = X_train.iloc[val_idx]

    y_fold_train = y_train.iloc[train_idx]
    y_fold_val = y_train.iloc[val_idx]

    model = XGBClassifier(**params)

    print("Training fold...")

    model.fit(
        X_fold_train,
        y_fold_train,
    )

    probabilities = model.predict_proba(
        X_fold_val
    )[:, 1]

    oof_predictions[val_idx] = probabilities

    roc_auc = roc_auc_score(
        y_fold_val,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_fold_val,
        probabilities,
    )

    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC : {pr_auc:.4f}")

    cv_results.append(
        {
            "fold": fold,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
        }
    )


# ============================================================
# CV Summary
# ============================================================

cv_df = pd.DataFrame(cv_results)

print("\n========================================")
print("CROSS-VALIDATION SUMMARY")
print("========================================")

print(cv_df.to_string(index=False))

print(
    f"\nMean ROC-AUC: "
    f"{cv_df['roc_auc'].mean():.4f}"
)

print(
    f"Mean PR-AUC : "
    f"{cv_df['pr_auc'].mean():.4f}"
)


# ============================================================
# Threshold Optimization
# ============================================================

print("\n========================================")
print("OOF THRESHOLD OPTIMIZATION")
print("========================================")

threshold_results = []

for threshold in np.arange(
    0.05,
    0.96,
    0.05,
):

    predictions = (
        oof_predictions >= threshold
    ).astype(int)

    precision = precision_score(
        y_train,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_train,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_train,
        predictions,
        zero_division=0,
    )

    accuracy = accuracy_score(
        y_train,
        predictions,
    )

    threshold_results.append(
        {
            "threshold": round(threshold, 2),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )


threshold_df = pd.DataFrame(
    threshold_results
)

print(
    "\nThreshold  Accuracy  Precision  Recall  F1"
)

print("---------------------------------------------")

for _, row in threshold_df.iterrows():

    print(
        f"{row['threshold']:.2f}"
        f"       "
        f"{row['accuracy']:.4f}"
        f"      "
        f"{row['precision']:.4f}"
        f"     "
        f"{row['recall']:.4f}"
        f"  "
        f"{row['f1']:.4f}"
    )


# ============================================================
# Select threshold
# ============================================================

best_row = threshold_df.loc[
    threshold_df["f1"].idxmax()
]

BEST_THRESHOLD = float(
    best_row["threshold"]
)

print("\nBest threshold based on F1:")

print(f"{BEST_THRESHOLD:.2f}")

print(
    f"Precision: "
    f"{best_row['precision']:.4f}"
)

print(
    f"Recall   : "
    f"{best_row['recall']:.4f}"
)

print(
    f"F1 Score : "
    f"{best_row['f1']:.4f}"
)


# ============================================================
# Train final production model
# ============================================================

print("\n========================================")
print("TRAINING FINAL PRODUCTION MODEL")
print("========================================")

final_model = XGBClassifier(**params)

final_model.fit(
    X_train,
    y_train,
)

print("Final model trained successfully.")


# ============================================================
# Final test evaluation
# ============================================================

test_probabilities = final_model.predict_proba(
    X_test
)[:, 1]

test_predictions = (
    test_probabilities >= BEST_THRESHOLD
).astype(int)


accuracy = accuracy_score(
    y_test,
    test_predictions,
)

precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0,
)

recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0,
)

f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0,
)

roc_auc = roc_auc_score(
    y_test,
    test_probabilities,
)

pr_auc = average_precision_score(
    y_test,
    test_probabilities,
)


# ============================================================
# Results
# ============================================================

print("\n========================================")
print("FINAL TEST SET RESULTS")
print("========================================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")
print(f"PR-AUC   : {pr_auc:.4f}")

print(
    f"\nProduction threshold: "
    f"{BEST_THRESHOLD:.2f}"
)


# ============================================================
# Save model
# ============================================================

print("\nSaving production model...")

MODEL_PATH = MODEL_DIR / "fraud_model.skops"

sio.dump(
    final_model,
    MODEL_PATH,
)

THRESHOLD_PATH = MODEL_DIR / "threshold.txt"

THRESHOLD_PATH.write_text(
    str(BEST_THRESHOLD)
)

print(f"Model saved: {MODEL_PATH}")
print(f"Threshold saved: {THRESHOLD_PATH}")
print(f"Mappings saved: {MAPPINGS_PATH}")

print("\n========================================")
print("TRAINING COMPLETE")
print("========================================")