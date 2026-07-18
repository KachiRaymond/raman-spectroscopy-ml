import pickle
import pandas as pd

from .preprocessing import process_folder


def load_model(model_file):

    with open(model_file, "rb") as f:

        artifact = pickle.load(f)

    return artifact["model"], artifact["scaler"]


def preprocess_for_model(df):

    sample_ids = df["sample_id"]

    X = df.drop(
        columns=["sample_id"]
    )

    return sample_ids, X


def predict_folder(
    folder_path,
    model_file
):

    spectra = process_folder(folder_path)

    sample_ids, X = preprocess_for_model(
        spectra
    )

    model, scaler = load_model(
        model_file
    )

    if scaler is not None:
        X = scaler.transform(X)

    predictions = model.predict(X)

    output = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "prediction": predictions
        }
    )

    return output


if __name__ == "__main__":

    predictions = predict_folder(
        "../data/raw/inference/24062026",
        "../models/trained/svm.pkl"
    )

    predictions.to_csv(
        "../data/predictions/inference_results.csv",
        index=False
    )

    print(predictions.head())