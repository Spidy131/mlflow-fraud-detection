import pandas as pd
import numpy as np
import skops.io as sio

from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

from xgboost import XGBClassifier


# ============================================================
# 1. Configuration
# ============================================================

DATA_PATH = Path("data/processed/fraud_baseline.csv")

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "fraud_model.skops"
THRESHOLD_PATH = MODEL_DIR / "threshold.txt"

RANDOM_STATE = 42

N_FOLDS = 5


# ============================================================
# 2. Load dataset
# ============================================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ============================================================
# 3. Features / target
# ============================================================

X = df.drop(columns=["isFraud"])
y = df["isFraud"]


# ============================================================
# 4. Feature types
# ============================================================

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numeric_features = X.select_dtypes(
    exclude=["object"]
).columns.tolist()

print()
print("Numeric features:", len(numeric_features))
print("Categorical features:", len(categorical_features))


# ============================================================
# 5. Preprocessing
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        ),
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        ),
    ]
)


# ============================================================
# 6. Train/Test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)


print()
print("Training rows:", len(X_train))
print("Testing rows :", len(X_test))


# ============================================================
# 7. Class imbalance
# ============================================================

negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

scale_pos_weight = negative / positive

print()
print("Class distribution")
print("-------------------")
print("Legitimate:", negative)
print("Fraud     :", positive)
print(
    f"scale_pos_weight: {scale_pos_weight:.4f}"
)


# ============================================================
# 8. Regularized XGBoost
# ============================================================

xgb_model = XGBClassifier(

    # Number of trees
    n_estimators=500,

    # Smaller learning rate + more trees
    learning_rate=0.03,

    # Reduced complexity
    max_depth=4,

    # Minimum loss reduction required for split
    gamma=0.10,

    # Minimum child weight
    min_child_weight=5,

    # L1 regularization
    reg_alpha=0.10,

    # L2 regularization
    reg_lambda=5.0,

    # Row subsampling
    subsample=0.80,

    # Feature subsampling
    colsample_bytree=0.80,

    # Handle class imbalance
    scale_pos_weight=scale_pos_weight,

    random_state=RANDOM_STATE,

    eval_metric="aucpr",

    n_jobs=-1
)


# ============================================================
# 9. Stratified K-Fold Cross Validation
# ============================================================

print()
print("========================================")
print("5-FOLD STRATIFIED CROSS VALIDATION")
print("========================================")


skf = StratifiedKFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE
)


oof_predictions = np.zeros(
    len(X_train)
)


cv_results = []


for fold, (train_idx, valid_idx) in enumerate(
    skf.split(X_train, y_train),
    start=1
):

    print()
    print(f"========== FOLD {fold} ==========")

    X_fold_train = X_train.iloc[train_idx]
    X_fold_valid = X_train.iloc[valid_idx]

    y_fold_train = y_train.iloc[train_idx]
    y_fold_valid = y_train.iloc[valid_idx]


    # ----------------------------------------
    # Create pipeline
    # ----------------------------------------

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                xgb_model
            )
        ]
    )


    print("Training fold...")

    pipeline.fit(
        X_fold_train,
        y_fold_train
    )


    # ----------------------------------------
    # Validation probabilities
    # ----------------------------------------

    probabilities = pipeline.predict_proba(
        X_fold_valid
    )[:, 1]


    oof_predictions[valid_idx] = probabilities


    # ----------------------------------------
    # Metrics
    # ----------------------------------------

    roc_auc = roc_auc_score(
        y_fold_valid,
        probabilities
    )

    pr_auc = average_precision_score(
        y_fold_valid,
        probabilities
    )


    cv_results.append(
        {
            "fold": fold,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc
        }
    )


    print(
        f"ROC-AUC: {roc_auc:.4f}"
    )

    print(
        f"PR-AUC : {pr_auc:.4f}"
    )


# ============================================================
# 10. CV summary
# ============================================================

cv_df = pd.DataFrame(cv_results)


print()
print("========================================")
print("CROSS-VALIDATION SUMMARY")
print("========================================")

print(
    cv_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


print()
print(
    f"Mean ROC-AUC: "
    f"{cv_df['roc_auc'].mean():.4f}"
)

print(
    f"Mean PR-AUC : "
    f"{cv_df['pr_auc'].mean():.4f}"
)


# ============================================================
# 11. OOF threshold optimization
# ============================================================

print()
print("========================================")
print("OOF THRESHOLD OPTIMIZATION")
print("========================================")


thresholds = np.arange(
    0.05,
    0.96,
    0.05
)


threshold_results = []


for threshold in thresholds:

    predictions = (
        oof_predictions >= threshold
    ).astype(int)


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
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    )


threshold_df = pd.DataFrame(
    threshold_results
)


print()
print(
    "Threshold  Precision  Recall  F1"
)

print(
    "--------------------------------"
)


for _, row in threshold_df.iterrows():

    print(
        f"{row['threshold']:.2f}"
        f"        "
        f"{row['precision']:.4f}"
        f"     "
        f"{row['recall']:.4f}"
        f"  "
        f"{row['f1']:.4f}"
    )


# ============================================================
# 12. Select best F1 threshold
# ============================================================

best_row = threshold_df.loc[
    threshold_df["f1"].idxmax()
]


BEST_THRESHOLD = float(
    best_row["threshold"]
)


print()
print("Best threshold:")
print(
    f"{BEST_THRESHOLD:.2f}"
)

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
# 13. Train final production model
# ============================================================

print()
print("========================================")
print("TRAINING FINAL PRODUCTION MODEL")
print("========================================")


final_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            xgb_model
        )
    ]
)


final_pipeline.fit(
    X_train,
    y_train
)


print(
    "Final model trained successfully."
)


# ============================================================
# 14. Final test predictions
# ============================================================

test_probabilities = final_pipeline.predict_proba(
    X_test
)[:, 1]


test_predictions = (
    test_probabilities >= BEST_THRESHOLD
).astype(int)


# ============================================================
# 15. Final metrics
# ============================================================

accuracy = (
    test_predictions == y_test.values
).mean()


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


print()
print("========================================")
print("FINAL TEST SET RESULTS")
print("========================================")


print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)

print(
    f"ROC-AUC   : {roc_auc:.4f}"
)

print(
    f"PR-AUC    : {pr_auc:.4f}"
)


# ============================================================
# 16. Save model
# ============================================================

print()
print("Saving production model...")


sio.dump(
    final_pipeline,
    MODEL_PATH
)


# ============================================================
# 17. Save threshold
# ============================================================

THRESHOLD_PATH.write_text(
    str(BEST_THRESHOLD)
)


print(
    f"Model saved: {MODEL_PATH}"
)

print(
    f"Threshold saved: {THRESHOLD_PATH}"
)


# ============================================================
# 18. Complete
# ============================================================

print()
print("========================================")
print("TRAINING COMPLETE")
print("========================================")

print(
    f"Production threshold: "
    f"{BEST_THRESHOLD:.2f}"
)

print(
    f"Production model: "
    f"{MODEL_PATH}"
)