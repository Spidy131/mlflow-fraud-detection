import pandas as pd
import numpy as np

from pathlib import Path

import mlflow
import mlflow.sklearn

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

from xgboost import XGBClassifier

import skops.io as sio


# ==================================================
# 1. Configuration
# ==================================================

DATA_PATH = Path(
    "data/processed/fraud_baseline.csv"
)

MODEL_PATH = Path(
    "models/fraud_model.skops"
)

THRESHOLD_PATH = Path(
    "models/threshold.txt"
)

EXPERIMENT_NAME = "fraud-detection"

RANDOM_STATE = 42


# ==================================================
# 2. Load dataset
# ==================================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ==================================================
# 3. Separate features and target
# ==================================================

X = df.drop(
    columns=["isFraud"]
)

y = df["isFraud"]


# ==================================================
# 4. Identify feature types
# ==================================================

categorical_features = X.select_dtypes(
    include=["object", "str"]
).columns.tolist()

numeric_features = X.select_dtypes(
    exclude=["object", "str"]
).columns.tolist()


print(
    "Numeric features:",
    len(numeric_features)
)

print(
    "Categorical features:",
    len(categorical_features)
)


# ==================================================
# 5. Preprocessing
# ==================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
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
        )
    ]
)


# ==================================================
# 6. Train/Test split
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)


print(
    "Training rows:",
    len(X_train)
)

print(
    "Testing rows :",
    len(X_test)
)


# ==================================================
# 7. Class imbalance
# ==================================================

negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_pos_weight = (
    negative_count / positive_count
)


print("\nClass distribution")
print("-------------------")
print(
    "Legitimate:",
    negative_count
)
print(
    "Fraud     :",
    positive_count
)
print(
    "scale_pos_weight:",
    round(scale_pos_weight, 4)
)


# ==================================================
# 8. Base XGBoost model
# ==================================================

xgb = XGBClassifier(

    objective="binary:logistic",

    eval_metric="aucpr",

    random_state=RANDOM_STATE,

    n_jobs=-1,

    scale_pos_weight=scale_pos_weight
)


# ==================================================
# 9. Complete pipeline
# ==================================================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            xgb
        )
    ]
)


# ==================================================
# 10. Hyperparameter search space
# ==================================================

param_distributions = {

    # Number of trees
    "model__n_estimators": [
        300,
        500,
        700,
        900
    ],

    # Tree complexity
    "model__max_depth": [
        3,
        4,
        5,
        6
    ],

    # Learning rate
    "model__learning_rate": [
        0.01,
        0.03,
        0.05,
        0.08,
        0.1
    ],

    # Minimum child weight
    "model__min_child_weight": [
        1,
        3,
        5,
        10
    ],

    # Minimum loss reduction
    "model__gamma": [
        0,
        0.05,
        0.1,
        0.3,
        0.5
    ],

    # Row sampling
    "model__subsample": [
        0.7,
        0.8,
        0.9,
        1.0
    ],

    # Feature sampling
    "model__colsample_bytree": [
        0.6,
        0.8,
        1.0
    ],

    # L1 regularization
    "model__reg_alpha": [
        0,
        0.01,
        0.1,
        0.5,
        1.0
    ],

    # L2 regularization
    "model__reg_lambda": [
        1,
        3,
        5,
        10,
        20
    ]
}


# ==================================================
# 11. Cross-validation
# ==================================================

print("\n========================================")
print("XGBOOST RANDOMIZED CV")
print("========================================")

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=RANDOM_STATE
)


search = RandomizedSearchCV(

    estimator=pipeline,

    param_distributions=param_distributions,

    n_iter=10,

    scoring="average_precision",

    cv=cv,

    random_state=RANDOM_STATE,

    verbose=2,

    n_jobs=1,

    return_train_score=True
)


# ==================================================
# 12. MLflow
# ==================================================

mlflow.set_experiment(
    EXPERIMENT_NAME
)


with mlflow.start_run(
    run_name="XGBoost-Regularized-RandomizedSearch"
):

    print("\nStarting hyperparameter search...")

    search.fit(
        X_train,
        y_train
    )

    print("\n========================================")
    print("BEST HYPERPARAMETERS")
    print("========================================")

    print(
        search.best_params_
    )

    print(
        "\nBest CV PR-AUC:",
        round(
            search.best_score_,
            4
        )
    )


    # ==================================================
    # 13. Log best parameters
    # ==================================================

    mlflow.log_param(
        "cv_folds",
        3
    )

    mlflow.log_param(
        "search_iterations",
        10
    )

    mlflow.log_param(
        "optimization_metric",
        "average_precision"
    )

    mlflow.log_params(
        {
            k: str(v)
            for k, v
            in search.best_params_.items()
        }
    )

    mlflow.log_metric(
        "best_cv_pr_auc",
        search.best_score_
    )


    # ==================================================
    # 14. Best model
    # ==================================================

    best_model = search.best_estimator_


    # ==================================================
    # 15. Test-set probability
    # ==================================================

    print("\n========================================")
    print("FINAL TEST SET EVALUATION")
    print("========================================")

    y_probability = (
        best_model.predict_proba(
            X_test
        )[:, 1]
    )


    # ==================================================
    # 16. Threshold optimization
    # ==================================================

    print("\n========================================")
    print("THRESHOLD OPTIMIZATION")
    print("========================================")

    best_threshold = 0.5
    best_f1 = 0

    threshold_results = []

    for threshold in np.arange(
        0.05,
        0.96,
        0.01
    ):

        y_pred_threshold = (
            y_probability >= threshold
        ).astype(int)

        precision = precision_score(
            y_test,
            y_pred_threshold,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred_threshold,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred_threshold,
            zero_division=0
        )

        threshold_results.append(
            {
                "threshold": round(
                    float(threshold),
                    2
                ),
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
        )

        if f1 > best_f1:

            best_f1 = f1
            best_threshold = threshold


    print(
        f"\nBest threshold: "
        f"{best_threshold:.2f}"
    )


    # ==================================================
    # 17. Final predictions
    # ==================================================

    y_pred = (
        y_probability >= best_threshold
    ).astype(int)


    # ==================================================
    # 18. Metrics
    # ==================================================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    pr_auc = average_precision_score(
        y_test,
        y_probability
    )


    # ==================================================
    # 19. Print metrics
    # ==================================================

    print("\n========================================")
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


    # ==================================================
    # 20. Confusion matrix
    # ==================================================

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print("\n========================================")
    print("CONFUSION MATRIX")
    print("========================================")

    print(cm)

    tn, fp, fn, tp = cm.ravel()

    print(
        f"\nTrue Negatives  : {tn}"
    )

    print(
        f"False Positives : {fp}"
    )

    print(
        f"False Negatives : {fn}"
    )

    print(
        f"True Positives  : {tp}"
    )


    # ==================================================
    # 21. Log final metrics
    # ==================================================

    mlflow.log_metric(
        "test_accuracy",
        accuracy
    )

    mlflow.log_metric(
        "test_precision",
        precision
    )

    mlflow.log_metric(
        "test_recall",
        recall
    )

    mlflow.log_metric(
        "test_f1",
        f1
    )

    mlflow.log_metric(
        "test_roc_auc",
        roc_auc
    )

    mlflow.log_metric(
        "test_pr_auc",
        pr_auc
    )

    mlflow.log_metric(
        "production_threshold",
        best_threshold
    )


    # ==================================================
    # 22. Save model
    # ==================================================

    print("\nSaving production model...")

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    THRESHOLD_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    # Save pipeline
    sio.dump(
        best_model,
        MODEL_PATH
    )


    # Save threshold
    THRESHOLD_PATH.write_text(
        str(best_threshold)
    )


    print(
        f"Model saved: {MODEL_PATH}"
    )

    print(
        f"Threshold saved: {THRESHOLD_PATH}"
    )


    # ==================================================
    # 23. Log model to MLflow
    # ==================================================

    mlflow.sklearn.log_model(
        best_model,
        name="model",
        skops_trusted_types=[
            "numpy.dtype",
            "sklearn.compose._column_transformer._RemainderColsList",
            "xgboost.core.Booster",
            "xgboost.sklearn.XGBClassifier"
        ]
    )


print("\n========================================")
print("TRAINING COMPLETE")
print("========================================")

print(
    f"Production threshold: "
    f"{best_threshold:.2f}"
)

print(
    f"Production model: "
    f"{MODEL_PATH}"
)

print(
    f"Threshold file: "
    f"{THRESHOLD_PATH}"
)