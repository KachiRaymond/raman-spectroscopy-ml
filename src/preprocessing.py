import os
import glob
import re
import pandas as pd
import numpy as np


def extract_id(filename):
    match = re.search(r'Scan\s+(\d+)', filename)
    return str(int(match.group(1))) if match else None


def read_raman_csv(filepath):
    try:
        df = pd.read_csv(filepath, header=None)
        if df.shape[1] < 2:
            raise ValueError("Invalid format")
        return df.iloc[:, 1].values  # only intensity
    except Exception as e:
        print(f"[ERROR] {filepath}: {e}")
        return None


def process_folder(folder_path):
    
    files = glob.glob(os.path.join(folder_path, "*.csv"))
     
    all_data = []
    max_len = 0

    # First pass: read + find max length
    temp = []
    for f in files:
        fname = os.path.basename(f)
        fid = extract_id(fname)

        if fid is None:
            continue

        intensity = read_raman_csv(f)
        if intensity is None:
            continue

        max_len = max(max_len, len(intensity))

        temp.append((fid, fname, intensity))

    print(f"[INFO] max spectrum length = {max_len}")

    # Second pass: pad + build rows
    for fid, fname, intensity in temp:
        padded = np.full(max_len, np.nan)
        padded[:len(intensity)] = intensity

        row = [fid, fname] + padded.tolist()
        all_data.append(row)

    # Build column names
    cols = ["id", "filename"] + [f"f{i+1}" for i in range(max_len)]

    df_final = pd.DataFrame(all_data, columns=cols)

    return df_final

if __name__ == "__main__":
     

    df = process_folder(path_to_data)

    print(df.shape)
    print(df.head())

    df.to_csv("./Raman_ML/data/raman_data_22_06_2026.csv", index=False)


def merge_ground_truth(df_spectra, gt_path):
    """
    Merge Raman wide dataframe with ground truth CSV (from file path)

    Parameters
    ----------
    df_spectra : pandas.DataFrame
        Output from process_folder()
        columns: id, filename, f1, f2, ...

    gt_path : str
        Path to ground truth CSV file

    Returns
    -------
    pandas.DataFrame
        Fully merged dataframe
    """

    # ---------- Load ground truth ----------
    df_gt = pd.read_csv(gt_path)

    # ---------- Fix ID formatting ----------
    # spectra IDs already clean (from earlier fix), but enforce
    df_spectra = df_spectra.copy()
    df_spectra["ID"] = df_spectra["id"].astype(str).str.strip()

    # ground truth IDs → remove any leading zeros just in case
    df_gt["ID"] = df_gt["ID"].astype(str).str.strip().apply(lambda x: str(int(x)))

    # ---------- Clean text fields ----------
    for col in df_gt.columns:
        if df_gt[col].dtype == object:
            df_gt[col] = (
                df_gt[col]
                .fillna("")
                .astype(str)
                .str.strip()
            )

    # replace "0" with empty (your encoding)
    component_cols = [
        "Component1", "Component2", "Component3",
        "Component4", "Component5", "Component6"
    ]

    for col in component_cols:
        if col in df_gt.columns:
            df_gt[col] = df_gt[col].replace("0", "")

    # ---------- Merge ----------
    df_merged = pd.merge(
        df_gt,
        df_spectra,
        on="ID",
        how="inner"
    )

    # optional: drop duplicate id column
    df_merged = df_merged.drop(columns=["id"])

    return df_merged
