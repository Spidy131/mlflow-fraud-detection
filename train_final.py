import pandas as pd
import numpy as np
import skops.io as sio

from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from xgboost import XGBClassifier


# ============================================================
# Configuration
# ============================================================

DATA_PATH = "data/processed/fraud_baseline.csv"

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "fraud_model.skops"
THRESHOLD_PATH = MODEL_DIR / "threshold.txt"

RANDOM_STATE = 42
N_SPLITS = 5


# ============================================================
# Load dataset
# ============================================================

print("Loading processed fraud dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# ============================================================
# Feature Engineering
# ============================================================

print("\nCreating fraud-related features...")


# Transaction amount transformations
df["TransactionAmt_log"] = np.log1p(
    df["TransactionAmt"]
)

df["TransactionAmt_sqrt"] = np.sqrt(
    df["TransactionAmt"].clip(lower=0)
)


# Card combinations
df["card1_card2"] = (
    df["card1"].astype(str)
    + "_"
    + df["card2"].astype(str)
)

df["card3_card5"] = (
    df["card3"].astype(str)
    + "_"
    + df["card5"].astype(str)
)


# Address combination
df["addr1_addr2"] = (
    df["addr1"].astype(str)
    + "_"
    + df["addr2"].astype(str)
)


# Email relationship
df["email_match"] = (
    df["P_emaildomain"].fillna("missing")
    ==
    df["R_emaildomain"].fillna("missing")
).astype(int)


# Device information
df["device_missing"] = (
    df["DeviceInfo"].isna()
).astype(int)

df["email_missing"] = (
    df["P_emaildomain"].isna()
).astype(int)


# C feature aggregation
c_columns = [
    f"C{i}" for i in range(1, 11)
    if f"C{i}" in df.columns
]

if c_columns:
    df["C_sum"] = df[c_columns].sum(axis=1)
    df["C_mean"] = df[c_columns].mean(axis=1)
    df["C_max"] = df[c_columns].max(axis=1)


# D feature aggregation
d_columns = [
    f"D{i}" for i in range(1, 5)
    if f"D{i}" in df.columns
]

if d_columns:
    df["D_sum"] = df[d_columns].sum(axis=1)
    df["D_mean"] = df[d_columns].mean(axis=1)
    df["D_max"] = df[d_columns].max(axis=1)


# M feature statistics
m_columns = [
    f"M{i}" for i in range(1, 10)
    if f"M{i}" in df.columns
]

if m_columns:
    df["M_missing_count"] = (
        df[m_columns]
        .isna()
        .sum(axis=1)
    )


# ============================================================
# Prepare X / y
# ============================================================

TARGET = "isFraud"

X = df.drop(columns=[TARGET])
y = df[TARGET]


categorical_features = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()


print(f"Numeric features: {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")


# ============================================================
# Handle categorical values
# ============================================================

for column in categorical_features:

    X[column] = (
        X[column]
        .fillna("missing")
        .astype("category")
    )


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

base_scale_pos_weight = negative / positive

print("\nClass distribution")
print("-------------------")
print(f"Legitimate: {negative}")
print(f"Fraud     : {positive}")
print(
    f"Base scale_pos_weight: "
    f"{base_scale_pos_weight:.4f}"
)


# ============================================================
# Final XGBoost parameters
# ============================================================

params = {

    "n_estimators": 1000,

    "max_depth": 6,

    "learning_rate": 0.05,

    "subsample": 0.85,

    "colsample_bytree": 0.85,

    # Regularization
    "reg_alpha": 0.2,

    "reg_lambda": 8.0,

    "min_child_weight": 3,

    # Classification
    "objective": "binary:logistic",

    "eval_metric": "aucpr",

    # Tuned imbalance weight
    "scale_pos_weight": 20.0,

    "tree_method": "hist",

    "enable_categorical": True,

    "random_state": RANDOM_STATE,

    "n_jobs": -1,

    "verbosity": 0,
}


# ============================================================
# 5-Fold OOF Cross Validation
# ============================================================

print("\n============================================================")
print("5-FOLD STRATIFIED CROSS VALIDATION")
print("============================================================")


skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)


oof_predictions = np.zeros(
    len(X_train)
)


cv_results = []


for fold, (train_idx, val_idx) in enumerate(
    skf.split(X_train, y_train),
    start=1,
):

    print(
        f"\n========== FOLD {fold} =========="
    )

    X_fold_train = X_train.iloc[train_idx]
    X_fold_val = X_train.iloc[val_idx]

    y_fold_train = y_train.iloc[train_idx]
    y_fold_val = y_train.iloc[val_idx]


    model = XGBClassifier(**params)


    print("Training fold...")


    model.fit(
        X_fold_train,
        y_fold_train,

        eval_set=[
            (X_fold_val, y_fold_val)
        ],

        verbose=False,
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


    print(
        f"ROC-AUC: {roc_auc:.4f}"
    )

    print(
        f"PR-AUC : {pr_auc:.4f}"
    )


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


print("\n============================================================")
print("CROSS-VALIDATION SUMMARY")
print("============================================================")


print(
    cv_df.to_string(index=False)
)


print(
    f"\nMean ROC-AUC: "
    f"{cv_df['roc_auc'].mean():.4f}"
)


print(
    f"Mean PR-AUC : "
    f"{cv_df['pr_auc'].mean():.4f}"
)


# ============================================================
# Fine Threshold Optimization
# ============================================================

print("\n============================================================")
print("FINE THRESHOLD OPTIMIZATION")
print("============================================================")


threshold_results = []


for threshold in np.arange(
    0.50,
    0.951,
    0.01,
):

    predictions = (
        oof_predictions >= threshold
    ).astype(int)


    accuracy = accuracy_score(
        y_train,
        predictions,
    )


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


    threshold_results.append(
        {
            "threshold": round(
                float(threshold),
                2
            ),

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

print(
    "---------------------------------------------"
)


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
# Best F1 threshold
# ============================================================

best_f1_row = threshold_df.loc[
    threshold_df["f1"].idxmax()
]


BEST_THRESHOLD = float(
    best_f1_row["threshold"]
)


print("\n========================================")
print("BEST F1 THRESHOLD")
print("========================================")


print(
    f"Threshold : {BEST_THRESHOLD:.2f}"
)

print(
    f"Precision : "
    f"{best_f1_row['precision']:.4f}"
)

print(
    f"Recall    : "
    f"{best_f1_row['recall']:.4f}"
)

print(
    f"F1 Score  : "
    f"{best_f1_row['f1']:.4f}"
)


# ============================================================
# Best threshold with precision >= 70%
# ============================================================

precision_constraint = 0.70


precision_candidates = threshold_df[
    threshold_df["precision"]
    >= precision_constraint
]


if len(precision_candidates) > 0:

    best_precision_row = (
        precision_candidates
        .sort_values(
            "recall",
            ascending=False
        )
        .iloc[0]
    )


    print(
        "\n========================================"
    )

    print(
        "BEST THRESHOLD WITH PRECISION >= 70%"
    )

    print(
        "========================================"
    )


    print(
        f"Threshold : "
        f"{best_precision_row['threshold']:.2f}"
    )

    print(
        f"Precision : "
        f"{best_precision_row['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{best_precision_row['recall']:.4f}"
    )

    print(
        f"F1 Score  : "
        f"{best_precision_row['f1']:.4f}"
    )


# ============================================================
# Train final production model
# ============================================================

print("\n============================================================")
print("TRAINING FINAL PRODUCTION MODEL")
print("============================================================")


final_model = XGBClassifier(
    **params
)


final_model.fit(
    X_train,
    y_train,

    eval_set=[
        (X_test, y_test)
    ],

    verbose=False,
)


print(
    "Final model trained successfully."
)


# ============================================================
# Final Test Prediction
# ============================================================

test_probabilities = (
    final_model
    .predict_proba(X_test)[:, 1]
)


test_predictions = (
    test_probabilities >= BEST_THRESHOLD
).astype(int)


# ============================================================
# Final Metrics
# ============================================================

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


cm = confusion_matrix(
    y_test,
    test_predictions,
)


# ============================================================
# Results
# ============================================================

print("\n============================================================")
print("FINAL TEST SET RESULTS")
print("============================================================")


print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)

print(
    f"ROC-AUC  : {roc_auc:.4f}"
)

print(
    f"PR-AUC   : {pr_auc:.4f}"
)


print(
    f"\nProduction threshold: "
    f"{BEST_THRESHOLD:.2f}"
)


print("\nConfusion Matrix:")

print(cm)


# ============================================================
# Save Model
# ============================================================

print("\nSaving production model...")


sio.dump(
    final_model,
    MODEL_PATH,
)


THRESHOLD_PATH.write_text(
    str(BEST_THRESHOLD)
)


print(
    f"Model saved: {MODEL_PATH}"
)


print(
    f"Threshold saved: {THRESHOLD_PATH}"
)


print("\n============================================================")
print("TRAINING COMPLETE")
print("============================================================")