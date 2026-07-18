import json
import pickle

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.config import(TRAINED_MODELS_DIR, RANDOM_STATE, TEST_SIZE)

# -------------------------------
# 1. Prepare data
# -------------------------------

def prepare_data(df):
    df = df.copy()

    # ---- clean column names ----
    df.columns = (
        df.columns
        .str.strip()
        .str.replace("\n", " ")
        .str.replace("\r", " ")
    )

    #print("\n[DEBUG] Columns:")
    #print(df.columns.tolist())

    # ---- detect target column ----
    target_candidates = [c for c in df.columns if "primary" in c.lower()]
    
    if len(target_candidates) == 0:
        raise ValueError("No Primary component column found")

    target_col = target_candidates[0]
    print(f"[INFO] Using target column: {target_col}")

    # ---- create binary target ----
    df["target"] = (
        df[target_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("paracetamol")
    ).astype(int)

    # ---- feature columns ----
    feature_cols = [c for c in df.columns if c.startswith("f")]

    print(f"[INFO] Number of features: {len(feature_cols)}")

    # ---- convert to numeric (safe) ----
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").values

    # ✅ instead of dropping → FIX NaNs
    nan_count = np.isnan(X).sum()
    #print(f"[INFO] Total NaNs in X: {nan_count}")

    # ---- replace NaNs with 0 ----
    X = np.nan_to_num(X, nan=0.0).astype(np.float32)

    y = df["target"].values

    #print(f"[INFO] Final usable samples: {len(y)}")

    # ---- normalize spectra ----
    max_vals = np.max(X, axis=1, keepdims=True)
    max_vals[max_vals == 0] = 1   # avoid divide by zero
    X = X / max_vals

    return X, y

# -------------------------------
# 2. Train-test split
# -------------------------------

def split_data(X, y):
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )


# -------------------------------
# 3. Train model
# -------------------------------

def train_model(model, X_train, y_train, scale=False):
    scaler = None

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

    model.fit(X_train, y_train)
    return model, scaler


# -------------------------------
# 4. Evaluate model
# -------------------------------

def evaluate_model(model, X_test, y_test, scaler=None):
    
    if scaler is not None:
        X_test = scaler.transform(X_test)

    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return cm


# -------------------------------
# 5. Pipeline runner
# -------------------------------

def run_pipeline(df, model, scale=False, model_name="model"):

    print(f"\n===== {model_name.upper()} =====")

    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    trained_model, scaler = train_model(model, X_train, y_train, scale)

    cm = evaluate_model(trained_model, X_test, y_test, scaler)

    # Save model and scaler
    model_bundle = {
        "model": trained_model,
        "scaler": scaler
    }

    TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_file = TRAINED_MODELS_DIR / f"{model_name.lower()}.pkl"

    with open(model_file, "wb") as f:
        pickle.dump(model_bundle, f)

    print(f"[INFO] Saved model to {model_file}")

    return trained_model, cm