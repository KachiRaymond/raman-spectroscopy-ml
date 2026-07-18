import json
import pickle

import pandas as pd

from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from .config import (
    RANDOM_STATE,
    TEST_SIZE
)


TARGET_COLUMN = "label"


def prepare_data(df):

    X = df.drop(
        columns=["sample_id", TARGET_COLUMN]
    )

    y = df[TARGET_COLUMN]

    return X, y


def split_data(X, y):

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )


def train_model(
    model,
    X_train,
    y_train,
    scale=False
):

    scaler = None

    if scale:

        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)

    model.fit(X_train, y_train)

    return model, scaler


def evaluate_model(
    model,
    X_test,
    y_test,
    scaler=None
):

    if scaler is not None:
        X_test = scaler.transform(X_test)

    preds = model.predict(X_test)

    results = {
        "accuracy": float(
            accuracy_score(y_test, preds)
        ),
        "classification_report":
            classification_report(
                y_test,
                preds,
                output_dict=True
            ),
        "confusion_matrix":
            confusion_matrix(
                y_test,
                preds
            ).tolist()
    }

    return results


def save_artifacts(
    model,
    scaler,
    model_path
):

    with open(model_path, "wb") as f:

        pickle.dump(
            {
                "model": model,
                "scaler": scaler
            },
            f
        )


def run_pipeline(
    df,
    model,
    model_name,
    scale=False,
    model_output=None,
    metrics_output=None
):

    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    model, scaler = train_model(
        model,
        X_train,
        y_train,
        scale
    )

    metrics = evaluate_model(
        model,
        X_test,
        y_test,
        scaler
    )

    if model_output:
        save_artifacts(
            model,
            scaler,
            model_output
        )

    if metrics_output:
        with open(metrics_output, "w") as f:
            json.dump(metrics, f, indent=4)

    return metrics


if __name__ == "__main__":

    df = pd.read_csv(
        "../data/processed/training_dataset.csv"
    )

    svm_metrics = run_pipeline(
        df=df,
        model=SVC(
            probability=True
        ),
        model_name="svm",
        scale=True,
        model_output="../models/trained/svm.pkl",
        metrics_output="../models/metrics/svm.json"
    )

    xgb_metrics = run_pipeline(
        df=df,
        model=XGBClassifier(
            random_state=42
        ),
        model_name="xgb",
        model_output="../models/trained/xgb.pkl",
        metrics_output="../models/metrics/xgb.json"
    )

    print(svm_metrics)
    print(xgb_metrics)