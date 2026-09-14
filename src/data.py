"""Loading and preprocessing -- same steps as the notebook."""
import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "SmartPhoneAddiction" / "data"

TARGET = "addicted_label"
ID = "id"
CAT = ["gender", "stress_level", "academic_work_impact"]   # non-numeric columns
N_PC = 7

TRAIN_FRAC = 0.8


def fit_prep(df, features):
    """Learn the preprocessing values from the training data."""
    p = {}

    # 1. values used to fill the missing data
    p["median"] = df[features].median()
    p["mode"] = df[CAT].mode().iloc[0]

    # 2. z-score:  (x - mean) / std
    z = df[features].fillna(p["median"])
    p["mean"] = z.mean()
    p["std"] = z.std()
    z = (z - p["mean"]) / p["std"]

    # 3. clip to +-3 standard deviations
    z = z.clip(-3, 3)

    # 4. PCA: eigenvectors of the covariance matrix, biggest variance first
    eigvals, eigvecs = np.linalg.eigh(np.cov(z.to_numpy(), rowvar=False))
    order = eigvals.argsort()[::-1]
    p["W"] = eigvecs[:, order][:, :N_PC]

    # 5. categories seen in training. The first one of each column is dropped:
    #    it is the default, and keeping it would repeat information.
    p["levels"] = {}
    for c in CAT:
        p["levels"][c] = sorted(df[c].dropna().unique())[1:]

    return p


def apply_prep(df, p, features):
    """Apply the learned values to any dataframe. -> (matrix, column names)"""
    z = df[features].fillna(p["median"])
    z = (z - p["mean"]) / p["std"]
    z = z.clip(-3, 3)

    pcs = z.to_numpy() @ p["W"]          # the 7 principal components

    columns = [f"PC{i+1}" for i in range(N_PC)]

    # one column per category: 1 if the row has that category, 0 if not
    dummies = []
    for c in CAT:
        filled = df[c].fillna(p["mode"][c])
        for level in p["levels"][c]:
            dummies.append((filled == level).to_numpy(dtype=float))
            columns.append(f"{c}_{level}")

    X = np.column_stack([pcs] + dummies)
    return X, columns


def split(df):
    """First 80% of train.csv for training, last 20% for validation."""
    n = int(len(df) * TRAIN_FRAC)
    return df.iloc[:n], df.iloc[n:]


def load():
    """-> {"train": (X, y), "val": (X, y), "test": X, "test_ids", "columns"}"""
    df = pd.read_csv(DATA / "train.csv")
    df_test = pd.read_csv(DATA / "test.csv")

    features = [c for c in df.select_dtypes("number").columns
                if c not in (ID, TARGET)]

    df_train, df_val = split(df)

    prep = fit_prep(df_train, features)          # train slice only, or it leaks
    X_train, columns = apply_prep(df_train, prep, features)
    X_val, _ = apply_prep(df_val, prep, features)
    X_test, _ = apply_prep(df_test, prep, features)

    return {
        "train": (X_train, df_train[TARGET].to_numpy(dtype=float)),
        "val": (X_val, df_val[TARGET].to_numpy(dtype=float)),
        "test": X_test,               # no labels: Kaggle scores this one
        "test_ids": df_test[ID],
        "columns": columns,
    }


def write_submission(ids, probs, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({ID: ids, TARGET: probs}).to_csv(path, index=False)
