import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from xgboost import XGBClassifier


# ==================================================
# 1. Configuration
# ==================================================

DATA_PATH = Path("data/processed/fraud_baseline.csv")

EXPERIMENT_NAME = "fraud-detection"


# ==================================================
# 2. Load dataset
# ==================================================

print("Loading processed fraud dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ==================================================
# 3. Separate features and target
# ==================================================

X = df.drop(columns=["isFraud"])
y = df["isFraud"]


# ==================================================
# 4. Identify feature types
# ==================================================

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numeric_features = X.select_dtypes(
    exclude=["object"]
).columns.tolist()


print("Numeric features:", len(numeric_features))
print("Categorical features:", len(categorical_features))


# ==================================================
# 5. Preprocessing pipelines
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


# ==================================================
# 6. Train/Test split
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("Training rows:", len(X_train))
print("Testing rows :", len(X_test))


# ==================================================
# 7. MLflow experiment
# ==================================================

mlflow.set_experiment(EXPERIMENT_NAME)


# ==================================================
# 8. Training function
# ==================================================

def train_model(model, model_name, model_type="sklearn"):

    print("\n========================================")
    print(f"Training: {model_name}")
    print("========================================")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    with mlflow.start_run(run_name=model_name):

        # ------------------------------------------
        # Log model parameters
        # ------------------------------------------

        mlflow.log_param(
            "model_name",
            model_name
        )

        mlflow.log_params(
            model.get_params()
        )

        # ------------------------------------------
        # Train
        # ------------------------------------------

        pipeline.fit(
            X_train,
            y_train
        )

        # ------------------------------------------
        # Predict
        # ------------------------------------------

        y_pred = pipeline.predict(X_test)

        y_probability = pipeline.predict_proba(
            X_test
        )[:, 1]

        # ------------------------------------------
        # Metrics
        # ------------------------------------------

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

        # ------------------------------------------
        # Log metrics
        # ------------------------------------------

        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        mlflow.log_metric(
            "precision",
            precision
        )

        mlflow.log_metric(
            "recall",
            recall
        )

        mlflow.log_metric(
            "f1_score",
            f1
        )

        mlflow.log_metric(
            "roc_auc",
            roc_auc
        )

        mlflow.log_metric(
            "pr_auc",
            pr_auc
        )

        # ------------------------------------------
        # Log model
        # ------------------------------------------

        if model_type == "xgboost":

            mlflow.sklearn.log_model(
                pipeline,
                name="model",
                skops_trusted_types=[
                    "numpy.dtype",
                    "sklearn.compose._column_transformer._RemainderColsList",
                    "xgboost.core.Booster",
                    "xgboost.sklearn.XGBClassifier",
                ],
            )

        else:

            mlflow.sklearn.log_model(
                pipeline,
                name="model",
                skops_trusted_types=[
                    "numpy.dtype",
                    "sklearn.compose._column_transformer._RemainderColsList",
                ],
            )

        # ------------------------------------------
        # Print results
        # ------------------------------------------

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"ROC-AUC  : {roc_auc:.4f}")
        print(f"PR-AUC   : {pr_auc:.4f}")


# ==================================================
# 9. Logistic Regression
# ==================================================

logistic_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

train_model(
    logistic_model,
    "Logistic Regression"
)


# ==================================================
# 10. Random Forest
# ==================================================

random_forest_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

train_model(
    random_forest_model,
    "Random Forest"
)


# ==================================================
# 11. XGBoost
# ==================================================

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1
)

train_model(
    xgb_model,
    "XGBoost",
    model_type="xgboost"
)