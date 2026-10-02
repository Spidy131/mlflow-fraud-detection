import pandas as pd
import numpy as np

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
# CONFIGURATION
# ============================================================

DATA_PATH = "data/processed/fraud_baseline.csv"

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "fraud_model.skops"
THRESHOLD_PATH = MODEL_DIR / "threshold.txt"

RANDOM_STATE = 42

N_SPLITS = 5

# Fine threshold search
THRESHOLD_START = 0.01
THRESHOLD_END = 0.99
THRESHOLD_STEP = 0.01


# ============================================================
# LOAD DATA
# ============================================================

print("Loading processed fraud dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# ============================================================
# TARGET
# ============================================================

TARGET = "isFraud"

X = df.drop(columns=[TARGET]).copy()
y = df[TARGET].copy()


# ============================================================
# FEATURE ENGINEERING
# ============================================================

print("\nCreating fraud-related features...")


# ------------------------------------------------------------
# Transaction amount
# ------------------------------------------------------------

X["TransactionAmt_log"] = np.log1p(
    X["TransactionAmt"].clip(lower=0)
)


# ------------------------------------------------------------
# C feature aggregates
# ------------------------------------------------------------

c_columns = [
    "C1", "C2", "C3", "C4", "C5",
    "C6", "C7", "C8", "C9", "C10"
]

existing_c = [
    col for col in c_columns
    if col in X.columns
]

X["C_sum"] = X[existing_c].sum(axis=1)
X["C_mean"] = X[existing_c].mean(axis=1)
X["C_max"] = X[existing_c].max(axis=1)

X["C_nonzero_count"] = (
    X[existing_c] != 0
).sum(axis=1)


# ------------------------------------------------------------
# D feature aggregates
# ------------------------------------------------------------

d_columns = [
    "D1", "D2", "D3", "D4"
]

existing_d = [
    col for col in d_columns
    if col in X.columns
]

X["D_sum"] = X[existing_d].sum(axis=1)
X["D_mean"] = X[existing_d].mean(axis=1)
X["D_max"] = X[existing_d].max(axis=1)

X["D_nonzero_count"] = (
    X[existing_d] != 0
).sum(axis=1)


# ------------------------------------------------------------
# Email relationship
# ------------------------------------------------------------

if (
    "P_emaildomain" in X.columns
    and "R_emaildomain" in X.columns
):

    X["email_match"] = (
        X["P_emaildomain"].fillna("missing")
        ==
        X["R_emaildomain"].fillna("missing")
    ).astype(int)


# ------------------------------------------------------------
# Missingness indicators
# ------------------------------------------------------------

missing_columns = [
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "dist1",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
    "DeviceInfo",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M7",
    "M8",
    "M9",
]

for column in missing_columns:

    if column in X.columns:

        X[f"{column}_missing"] = (
            X[column].isna()
        ).astype(int)


# ============================================================
# CATEGORICAL FEATURES
# ============================================================

categorical_features = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()


print(
    f"Numeric features: {len(numeric_features)}"
)

print(
    f"Categorical features: {len(categorical_features)}"
)


# ============================================================
# HANDLE CATEGORICAL VALUES
# ============================================================

for column in categorical_features:

    X[column] = (
        X[column]
        .fillna("missing")
        .astype("category")
    )


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE,
)


print(
    f"Training rows: {len(X_train)}"
)

print(
    f"Testing rows : {len(X_test)}"
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

negative = int(
    (y_train == 0).sum()
)

positive = int(
    (y_train == 1).sum()
)

base_scale_pos_weight = (
    negative / positive
)


print("\nClass distribution")
print("-------------------")

print(
    f"Legitimate: {negative}"
)

print(
    f"Fraud     : {positive}"
)

print(
    f"Base scale_pos_weight: "
    f"{base_scale_pos_weight:.4f}"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

# We don't use every possible value because
# 5-fold CV × many combinations can become expensive.

scale_weights = [
    20.0,
    25.0,
    base_scale_pos_weight,
    30.0,
    35.0,
]

candidate_params = [

    {
        "max_depth": 4,
        "min_child_weight": 3,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
    },

    {
        "max_depth": 5,
        "min_child_weight": 3,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
    },

    {
        "max_depth": 6,
        "min_child_weight": 3,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
    },

    {
        "max_depth": 5,
        "min_child_weight": 5,
        "learning_rate": 0.03,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
    },

    {
        "max_depth": 6,
        "min_child_weight": 5,
        "learning_rate": 0.03,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
    },
]


# ============================================================
# STRATIFIED CV
# ============================================================

print("\n")
print("=" * 60)
print("HYPERPARAMETER / SCALE WEIGHT SEARCH")
print("=" * 60)


# To keep runtime reasonable, use 3-fold CV during
# candidate selection.

tuning_skf = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=RANDOM_STATE,
)


best_score = -1
best_params = None


for candidate_number, candidate in enumerate(
    candidate_params,
    start=1,
):

    for weight in scale_weights:

        params = {
            "n_estimators": 1000,

            "max_depth": candidate["max_depth"],

            "learning_rate": candidate[
                "learning_rate"
            ],

            "subsample": candidate[
                "subsample"
            ],

            "colsample_bytree": candidate[
                "colsample_bytree"
            ],

            # Regularization
            "reg_alpha": 0.2,
            "reg_lambda": 8.0,
            "min_child_weight": candidate[
                "min_child_weight"
            ],

            "objective": "binary:logistic",

            "eval_metric": "aucpr",

            "scale_pos_weight": weight,

            "tree_method": "hist",

            "enable_categorical": True,

            "random_state": RANDOM_STATE,

            "n_jobs": -1,

            # Reduce output
            "verbosity": 0,
        }


        fold_scores = []


        print(
            f"\nCandidate {candidate_number} | "
            f"scale_pos_weight={weight:.2f} | "
            f"depth={candidate['max_depth']} | "
            f"min_child_weight={candidate['min_child_weight']}"
        )


        for fold, (
            train_idx,
            val_idx
        ) in enumerate(
            tuning_skf.split(
                X_train,
                y_train
            ),
            start=1
        ):

            X_fold_train = X_train.iloc[
                train_idx
            ]

            X_fold_val = X_train.iloc[
                val_idx
            ]

            y_fold_train = y_train.iloc[
                train_idx
            ]

            y_fold_val = y_train.iloc[
                val_idx
            ]


            model = XGBClassifier(
                **params
            )


            model.fit(
                X_fold_train,
                y_fold_train,

                eval_set=[
                    (
                        X_fold_val,
                        y_fold_val
                    )
                ],

                verbose=False,
            )


            probabilities = (
                model
                .predict_proba(
                    X_fold_val
                )[:, 1]
            )


            pr_auc = average_precision_score(
                y_fold_val,
                probabilities
            )

            fold_scores.append(
                pr_auc
            )


        mean_pr_auc = np.mean(
            fold_scores
        )


        print(
            f"Mean PR-AUC: "
            f"{mean_pr_auc:.4f}"
        )


        if mean_pr_auc > best_score:

            best_score = mean_pr_auc

            best_params = params.copy()


            print(
                "*** NEW BEST MODEL ***"
            )


print("\n")
print("=" * 60)
print("BEST PARAMETERS")
print("=" * 60)

print(
    f"Best CV PR-AUC: "
    f"{best_score:.4f}"
)

print(
    best_params
)


# ============================================================
# FINAL 5-FOLD OOF MODEL
# ============================================================

print("\n")
print("=" * 60)
print("FINAL 5-FOLD CROSS VALIDATION")
print("=" * 60)


skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)


oof_predictions = np.zeros(
    len(X_train)
)

cv_results = []


for fold, (
    train_idx,
    val_idx
) in enumerate(
    skf.split(
        X_train,
        y_train
    ),
    start=1
):

    print(
        f"\n========== FOLD {fold} =========="
    )

    X_fold_train = X_train.iloc[
        train_idx
    ]

    X_fold_val = X_train.iloc[
        val_idx
    ]

    y_fold_train = y_train.iloc[
        train_idx
    ]

    y_fold_val = y_train.iloc[
        val_idx
    ]


    model = XGBClassifier(
        **best_params
    )


    print("Training fold...")


    model.fit(
        X_fold_train,
        y_fold_train,

        eval_set=[
            (
                X_fold_val,
                y_fold_val
            )
        ],

        verbose=False,
    )


    probabilities = (
        model
        .predict_proba(
            X_fold_val
        )[:, 1]
    )


    oof_predictions[
        val_idx
    ] = probabilities


    roc_auc = roc_auc_score(
        y_fold_val,
        probabilities
    )

    pr_auc = average_precision_score(
        y_fold_val,
        probabilities
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
# CV SUMMARY
# ============================================================

cv_df = pd.DataFrame(
    cv_results
)


print("\n")
print("=" * 60)
print("CROSS-VALIDATION SUMMARY")
print("=" * 60)

print(
    cv_df.to_string(
        index=False
    )
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
# FINE THRESHOLD OPTIMIZATION
# ============================================================

print("\n")
print("=" * 60)
print("FINE THRESHOLD OPTIMIZATION")
print("=" * 60)


threshold_results = []


thresholds = np.arange(
    THRESHOLD_START,
    THRESHOLD_END + THRESHOLD_STEP,
    THRESHOLD_STEP,
)


for threshold in thresholds:

    predictions = (
        oof_predictions >= threshold
    ).astype(int)


    accuracy = accuracy_score(
        y_train,
        predictions
    )

    precision = precision_score(
        y_train,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_train,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_train,
        predictions,
        zero_division=0
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


# ============================================================
# BEST F1
# ============================================================

best_f1_row = threshold_df.loc[
    threshold_df["f1"].idxmax()
]


BEST_THRESHOLD = float(
    best_f1_row["threshold"]
)


print("\nBest threshold based on F1:")

print(
    f"{BEST_THRESHOLD:.2f}"
)

print(
    f"Precision: "
    f"{best_f1_row['precision']:.4f}"
)

print(
    f"Recall   : "
    f"{best_f1_row['recall']:.4f}"
)

print(
    f"F1 Score : "
    f"{best_f1_row['f1']:.4f}"
)


# ============================================================
# PRECISION-CONSTRAINED THRESHOLD
# ============================================================

# Business-oriented option:
# Find maximum recall while keeping precision >= 70%.

precision_target = 0.70


precision_constrained = threshold_df[
    threshold_df["precision"]
    >= precision_target
]


if len(precision_constrained) > 0:

    constrained_row = (
        precision_constrained
        .loc[
            precision_constrained[
                "recall"
            ].idxmax()
        ]
    )


    print("\n")
    print(
        "Best threshold with "
        f"precision >= {precision_target:.0%}:"
    )

    print(
        f"Threshold: "
        f"{constrained_row['threshold']:.2f}"
    )

    print(
        f"Precision: "
        f"{constrained_row['precision']:.4f}"
    )

    print(
        f"Recall   : "
        f"{constrained_row['recall']:.4f}"
    )

    print(
        f"F1 Score : "
        f"{constrained_row['f1']:.4f}"
    )


# ============================================================
# TRAIN FINAL PRODUCTION MODEL
# ============================================================

print("\n")
print("=" * 60)
print("TRAINING FINAL PRODUCTION MODEL")
print("=" * 60)


final_model = XGBClassifier(
    **best_params
)


final_model.fit(
    X_train,
    y_train,

    eval_set=[
        (
            X_test,
            y_test
        )
    ],

    verbose=False,
)


print(
    "Final model trained successfully."
)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

test_probabilities = (
    final_model
    .predict_proba(
        X_test
    )[:, 1]
)


test_predictions = (
    test_probabilities >= BEST_THRESHOLD
).astype(int)


accuracy = accuracy_score(
    y_test,
    test_predictions
)

precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    test_probabilities
)

pr_auc = average_precision_score(
    y_test,
    test_probabilities
)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("FINAL TEST SET RESULTS")
print("=" * 60)

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


# ============================================================
# SAVE MODEL
# ============================================================

print("\nSaving production model...")


sio.dump(
    final_model,
    MODEL_PATH
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


print("\n")
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)