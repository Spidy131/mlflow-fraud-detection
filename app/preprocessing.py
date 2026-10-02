import numpy as np
import pandas as pd


def create_features(data, training_categories=None):
    """
    Create the exact 74 features used during model training.

    Parameters
    ----------
    data : dict or pandas.DataFrame
        Transaction data.

    training_categories : dict, optional
        Training-time categorical values from XGBoost.
        If supplied, categorical columns are aligned to these
        categories so unseen API values become NaN instead of
        causing an XGBoost categorical error.

    Returns
    -------
    pandas.DataFrame
        Exactly 74 model features in training order.
    """

    # =========================================================
    # Convert input to DataFrame
    # =========================================================

    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = data.copy()
        # =========================================================
    # Ensure numeric columns have numeric dtype
    # =========================================================

    numeric_features = [
        "TransactionAmt",
        "card1",
        "card2",
        "card3",
        "card5",
        "addr1",
        "addr2",
        "dist1",
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7",
        "C8",
        "C9",
        "C10",
        "D1",
        "D2",
        "D3",
        "D4",
    ]

    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # =========================================================
    # Transaction amount features
    # =========================================================

    df["TransactionAmt_log"] = np.log1p(
        df["TransactionAmt"]
    )

    df["TransactionAmt_sqrt"] = np.sqrt(
        df["TransactionAmt"].clip(lower=0)
    )

    # =========================================================
    # Combination features
    # =========================================================

    df["card1_card2"] = (
        df["card1"].fillna(-1).astype(str)
        + "_"
        + df["card2"].fillna(-1).astype(str)
    )

    df["card3_card5"] = (
        df["card3"].fillna(-1).astype(str)
        + "_"
        + df["card5"].fillna(-1).astype(str)
    )

    df["addr1_addr2"] = (
        df["addr1"].fillna(-1).astype(str)
        + "_"
        + df["addr2"].fillna(-1).astype(str)
    )

    # =========================================================
    # Email matching
    # =========================================================

    df["email_match"] = (
        df["P_emaildomain"].fillna("missing")
        == df["R_emaildomain"].fillna("missing")
    ).astype(int)

    # =========================================================
    # Missing-value indicators
    # =========================================================

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

    for col in missing_columns:
        df[f"{col}_missing"] = (
            df[col].isna().astype(int)
        )

    # =========================================================
    # C features
    # =========================================================

    c_cols = [
        "C1", "C2", "C3", "C4", "C5",
        "C6", "C7", "C8", "C9", "C10"
    ]

    df["C_sum"] = df[c_cols].sum(axis=1)
    df["C_mean"] = df[c_cols].mean(axis=1)
    df["C_max"] = df[c_cols].max(axis=1)
    df["C_nonzero_count"] = (
        (df[c_cols] != 0).sum(axis=1)
    )

    # =========================================================
    # D features
    # =========================================================

    d_cols = ["D1", "D2", "D3", "D4"]

    df["D_sum"] = df[d_cols].sum(axis=1)
    df["D_mean"] = df[d_cols].mean(axis=1)
    df["D_max"] = df[d_cols].max(axis=1)
    df["D_nonzero_count"] = (
        (df[d_cols] != 0).sum(axis=1)
    )

    # =========================================================
    # M missing count
    # =========================================================

    m_cols = [
        "M1", "M2", "M3", "M4", "M5",
        "M6", "M7", "M8", "M9"
    ]

    df["M_missing_count"] = (
        df[m_cols].isna().sum(axis=1)
    )

    # =========================================================
    # EXACT 74 FEATURES
    # =========================================================

    feature_columns = [

        # Base features
        "TransactionAmt",
        "ProductCD",
        "card1",
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

        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7",
        "C8",
        "C9",
        "C10",

        "D1",
        "D2",
        "D3",
        "D4",

        "M1",
        "M2",
        "M3",
        "M4",
        "M5",
        "M6",
        "M7",
        "M8",
        "M9",

        # Engineered
        "TransactionAmt_log",
        "TransactionAmt_sqrt",
        "card1_card2",
        "card3_card5",
        "addr1_addr2",
        "email_match",

        # Missing indicators
        "card2_missing",
        "card3_missing",
        "card4_missing",
        "card5_missing",
        "card6_missing",
        "addr1_missing",
        "addr2_missing",
        "dist1_missing",
        "P_emaildomain_missing",
        "R_emaildomain_missing",
        "DeviceType_missing",
        "DeviceInfo_missing",
        "M1_missing",
        "M2_missing",
        "M3_missing",
        "M4_missing",
        "M5_missing",
        "M6_missing",
        "M7_missing",
        "M8_missing",
        "M9_missing",

        # C aggregates
        "C_sum",
        "C_mean",
        "C_max",
        "C_nonzero_count",

        # D aggregates
        "D_sum",
        "D_mean",
        "D_max",
        "D_nonzero_count",

        # M aggregate
        "M_missing_count",
    ]

    # =========================================================
    # Validate feature count
    # =========================================================

    if len(feature_columns) != 74:
        raise ValueError(
            f"Expected 74 features, "
            f"got {len(feature_columns)}"
        )

    df = df[feature_columns].copy()

    # =========================================================
    # Categorical columns
    # =========================================================

    categorical_features = [
        "ProductCD",
        "card4",
        "card6",
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
        "card1_card2",
        "card3_card5",
        "addr1_addr2",
    ]

    # =========================================================
    # Align with training categories
    # =========================================================

    for col in categorical_features:

        if training_categories is not None and col in training_categories:

            categories = training_categories[col]

            # Only use category alignment when training
            # categories are available.
            df[col] = pd.Categorical(
                df[col],
                categories=categories
            )

        else:
            df[col] = df[col].astype("category")

    # =========================================================
    # Final validation
    # =========================================================

    if df.shape[1] != 74:
        raise ValueError(
            f"Model input has {df.shape[1]} features; "
            f"expected exactly 74."
        )

    return df