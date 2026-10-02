import pandas as pd
import numpy as np
import mlflow
from pathlib import Path
from xgboost import XGBClassifier
import mlflow.xgboost

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

RANDOM_STATE = 42
N_SPLITS = 5

BEST_THRESHOLD = 0.78


# ============================================================
# MLflow Configuration
# ============================================================


mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("fraud-detection")


# ============================================================
# Start MLflow Run
# ============================================================

with mlflow.start_run(run_name="XGBoost_Fraud_Production"):

    print("Loading processed fraud dataset...")

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset shape: {df.shape}")


    # ========================================================
    # Prepare X / y
    # ========================================================

    TARGET = "isFraud"

    X = df.drop(columns=[TARGET])
    y = df[TARGET]


    # ========================================================
    # Feature Engineering
    # ========================================================

    print("\nCreating fraud-related features...")

    # Transaction amount transformations
    X["TransactionAmt_log"] = np.log1p(X["TransactionAmt"])
    X["TransactionAmt_sqrt"] = np.sqrt(X["TransactionAmt"])


    # Card interaction
    X["card1_card2"] = (
        X["card1"].fillna(-1).astype(str)
        + "_"
        + X["card2"].fillna(-1).astype(str)
    )


    X["card3_card5"] = (
        X["card3"].fillna(-1).astype(str)
        + "_"
        + X["card5"].fillna(-1).astype(str)
    )


    # Address interaction
    X["addr1_addr2"] = (
        X["addr1"].fillna(-1).astype(str)
        + "_"
        + X["addr2"].fillna(-1).astype(str)
    )


    # Email matching
    X["email_match"] = (
        X["P_emaildomain"].fillna("missing")
        == X["R_emaildomain"].fillna("missing")
    ).astype(int)


    # Missing value features
    X["card2_missing"] = X["card2"].isna().astype(int)
    X["card3_missing"] = X["card3"].isna().astype(int)
    X["card4_missing"] = X["card4"].isna().astype(int)
    X["card5_missing"] = X["card5"].isna().astype(int)
    X["card6_missing"] = X["card6"].isna().astype(int)

    X["addr1_missing"] = X["addr1"].isna().astype(int)
    X["addr2_missing"] = X["addr2"].isna().astype(int)
    X["dist1_missing"] = X["dist1"].isna().astype(int)

    X["P_emaildomain_missing"] = (
        X["P_emaildomain"].isna().astype(int)
    )

    X["R_emaildomain_missing"] = (
        X["R_emaildomain"].isna().astype(int)
    )

    X["DeviceType_missing"] = (
        X["DeviceType"].isna().astype(int)
    )

    X["DeviceInfo_missing"] = (
        X["DeviceInfo"].isna().astype(int)
    )


    # M columns missing
    for column in [
        "M1", "M2", "M3", "M4", "M5",
        "M6", "M7", "M8", "M9"
    ]:
        X[f"{column}_missing"] = (
            X[column].isna().astype(int)
        )


    # C features
    C_columns = [
        "C1", "C2", "C3", "C4", "C5",
        "C6", "C7", "C8", "C9", "C10"
    ]

    X["C_sum"] = X[C_columns].sum(axis=1)
    X["C_mean"] = X[C_columns].mean(axis=1)
    X["C_max"] = X[C_columns].max(axis=1)
    X["C_nonzero_count"] = (
        (X[C_columns] != 0).sum(axis=1)
    )


    # D features
    D_columns = [
        "D1", "D2", "D3", "D4"
    ]

    X["D_sum"] = X[D_columns].sum(axis=1)
    X["D_mean"] = X[D_columns].mean(axis=1)
    X["D_max"] = X[D_columns].max(axis=1)
    X["D_nonzero_count"] = (
        (X[D_columns] != 0).sum(axis=1)
    )


    # M missing count
    M_columns = [
        "M1", "M2", "M3", "M4", "M5",
        "M6", "M7", "M8", "M9"
    ]

    X["M_missing_count"] = X[M_columns].isna().sum(axis=1)


    # ========================================================
    # Categorical columns
    # ========================================================

    categorical_features = X.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")


    for column in categorical_features:
        X[column] = (
            X[column]
            .fillna("missing")
            .astype("category")
        )


    # ========================================================
    # Train / Test Split
    # ========================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows : {len(X_test)}")


    # ========================================================
    # Class imbalance
    # ========================================================

    negative = (y_train == 0).sum()
    positive = (y_train == 1).sum()

    scale_pos_weight = negative / positive

    print("\nClass distribution")
    print("-------------------")
    print(f"Legitimate: {negative}")
    print(f"Fraud     : {positive}")
    print(f"Base scale_pos_weight: {scale_pos_weight:.4f}")


    # ========================================================
    # XGBoost Parameters
    # ========================================================

    params = {
        "n_estimators": 1000,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,

        "reg_alpha": 0.2,
        "reg_lambda": 8.0,
        "min_child_weight": 3,

        "objective": "binary:logistic",
        "eval_metric": "aucpr",

        "scale_pos_weight": 20.0,

        "tree_method": "hist",
        "enable_categorical": True,

        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": 0,
    }


    # ========================================================
    # Log Parameters
    # ========================================================

    mlflow.log_params(params)

    mlflow.log_param(
        "dataset_rows",
        len(df)
    )

    mlflow.log_param(
        "dataset_features_original",
        df.shape[1] - 1
    )

    mlflow.log_param(
        "features_after_engineering",
        X.shape[1]
    )

    mlflow.log_param(
        "cv_folds",
        N_SPLITS
    )

    mlflow.log_param(
        "production_threshold",
        BEST_THRESHOLD
    )


    # ========================================================
    # Cross Validation
    # ========================================================

    print("\n============================================================")
    print("5-FOLD STRATIFIED CROSS VALIDATION")
    print("============================================================")

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


        mlflow.log_metric(
            f"fold_{fold}_roc_auc",
            roc_auc
        )

        mlflow.log_metric(
            f"fold_{fold}_pr_auc",
            pr_auc
        )


        cv_results.append(
            {
                "fold": fold,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
            }
        )


    # ========================================================
    # CV Summary
    # ========================================================

    cv_df = pd.DataFrame(cv_results)

    mean_roc_auc = cv_df["roc_auc"].mean()
    mean_pr_auc = cv_df["pr_auc"].mean()

    print("\n============================================================")
    print("CROSS-VALIDATION SUMMARY")
    print("============================================================")

    print(cv_df.to_string(index=False))

    print(
        f"\nMean ROC-AUC: {mean_roc_auc:.4f}"
    )

    print(
        f"Mean PR-AUC : {mean_pr_auc:.4f}"
    )


    mlflow.log_metric(
        "cv_mean_roc_auc",
        mean_roc_auc
    )

    mlflow.log_metric(
        "cv_mean_pr_auc",
        mean_pr_auc
    )


    # ========================================================
    # Final Production Model
    # ========================================================

    print("\n============================================================")
    print("TRAINING FINAL PRODUCTION MODEL")
    print("============================================================")

    final_model = XGBClassifier(**params)

    final_model.fit(
        X_train,
        y_train,
    )

    print("Final model trained successfully.")


    # ========================================================
    # Test Prediction
    # ========================================================

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

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        test_predictions
    ).ravel()


    # ========================================================
    # Results
    # ========================================================

    print("\n============================================================")
    print("FINAL TEST SET RESULTS")
    print("============================================================")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"PR-AUC   : {pr_auc:.4f}")

    print("\nConfusion Matrix")
    print("----------------")
    print(f"TN: {tn}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"TP: {tp}")


    # ========================================================
    # Log Final Metrics
    # ========================================================

    mlflow.log_metrics({
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1,
        "test_roc_auc": roc_auc,
        "test_pr_auc": pr_auc,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    })


    # ========================================================
    # Save Model
    # ========================================================

    print("\nSaving production model...")

    import skops.io as sio

    MODEL_PATH = MODEL_DIR / "fraud_model.skops"

    sio.dump(
        final_model,
        MODEL_PATH,
    )


    THRESHOLD_PATH = MODEL_DIR / "threshold.txt"

    THRESHOLD_PATH.write_text(
        str(BEST_THRESHOLD)
    )


    print(
        f"Model saved: {MODEL_PATH}"
    )

    print(
        f"Threshold saved: {THRESHOLD_PATH}"
    )


    # ========================================================
    # Log Artifacts
    # ========================================================

  # ============================================================
# LOG MODEL TO MLFLOW
# ============================================================

    print("Logging XGBoost model to MLflow...")

    mlflow.xgboost.log_model(
        final_model,
        name="model"
    )

    # Keep threshold as a separate artifact
    mlflow.log_artifact(
        str(THRESHOLD_PATH),
        artifact_path="model"
    )

    print("XGBoost model logged successfully.")

    # ========================================================
    # Tags
    # ========================================================

    mlflow.set_tag(
        "model_type",
        "XGBoost"
    )

    mlflow.set_tag(
        "problem_type",
        "Fraud Detection"
    )

    mlflow.set_tag(
        "deployment_status",
        "production_candidate"
    )

    mlflow.set_tag(
        "feature_engineering",
        "enabled"
    )


    print("\n============================================================")
    print("MLFLOW RUN COMPLETE")
    print("============================================================")

    print(
        f"Run ID: {mlflow.active_run().info.run_id}"
    )

    print(
        "\nMLflow tracking directory: ./mlruns"
    )