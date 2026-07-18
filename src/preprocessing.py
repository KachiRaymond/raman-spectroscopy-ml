import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import GROUND_TRUTH_FILE


def extract_id(filename):
    """
    Supports:

    Scan 2137.csv
    2137.csv
    2137_1.csv
    """

    stem = Path(filename).stem

    match = re.search(r"(\d+)", stem)

    if match:
        return str(int(match.group(1)))

    return stem


def read_raman_csv(filepath):

    df = pd.read_csv(filepath, header=None)

    if df.shape[1] < 2:
        raise ValueError(f"Invalid Raman file: {filepath}")

    return df.iloc[:, 1].values


def process_folder(folder_path):

    folder_path = Path(folder_path)

    spectra = []

    csv_files = sorted(folder_path.glob("*.csv"))

    for file in csv_files:

        try:

            intensity = read_raman_csv(file)

            sample_id = extract_id(file.name)

            row = {
                "sample_id": sample_id
            }

            for i, value in enumerate(intensity):
                row[f"f_{i}"] = value

            spectra.append(row)

        except Exception as e:
            print(f"Failed: {file.name}")
            print(e)

    return pd.DataFrame(spectra)


def merge_ground_truth(
    spectra_df,
    gt_path=GROUND_TRUTH_FILE
):

    gt = pd.read_csv(gt_path)

    gt["sample_id"] = gt["sample_id"].astype(str)

    spectra_df["sample_id"] = spectra_df["sample_id"].astype(str)

    merged = spectra_df.merge(
        gt,
        on="sample_id",
        how="inner"
    )

    return merged


def save_processed_data(
    training_folder,
    output_file,
    gt_path=GROUND_TRUTH_FILE
):

    spectra_df = process_folder(training_folder)

    final_df = merge_ground_truth(
        spectra_df,
        gt_path
    )

    final_df.to_csv(
        output_file,
        index=False
    )

    return final_df