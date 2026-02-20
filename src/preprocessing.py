import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
 
 
def preprocess_data(df: pd.DataFrame):
     print(f"Dataset shape: {df.shape}")
     print(f"Total missing values: {int(df.isnull().sum().sum())}")
     y = df["SalePrice"].copy()
     X = df.drop(columns=["SalePrice"])
     numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
     categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
     if numeric_cols:
         X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
     for col in categorical_cols:
         mode_vals = X[col].mode(dropna=True)
         fill_val = mode_vals.iloc[0] if len(mode_vals) else "Unknown"
         X[col] = X[col].fillna(fill_val)
     if categorical_cols:
         X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
     if numeric_cols:
         scaler = StandardScaler()
         X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
     return X, y
 
 
def _auto_find_dataset():

    candidates = [
        "data/raw/train.csv",
        "data/raw/data(1).csv",
        "data/raw/Housing(1).csv",
        "data/data(1).csv",
        "data/Housing(1).csv",
        "Housing(1).csv",
        "data(1).csv",
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    for pattern in ["**/data(1).csv", "**/Housing(1).csv", "**/train.csv"]:
        found = glob.glob(pattern, recursive=True)
        if found:
            return found[0]
    raise FileNotFoundError("Dataset not found. Place CSV in data/raw/ e.g. train.csv or Housing(1).csv")


def _normalize_target_column(df: pd.DataFrame):
    cols_lower = {c.lower(): c for c in df.columns}
    if "saleprice" in cols_lower:
        return df, cols_lower["saleprice"]
    if "medv" in cols_lower:
        df = df.rename(columns={cols_lower["medv"]: "SalePrice"})
        return df, "SalePrice"
    if "price" in cols_lower:
        df = df.rename(columns={cols_lower["price"]: "SalePrice"})
        return df, "SalePrice"
    raise KeyError("Target column not found. Expected one of: SalePrice, MEDV, Price")


def load_data(path: str | None = None):
    csv_path = path or _auto_find_dataset()
    df = pd.read_csv(csv_path)
    df, target_col = _normalize_target_column(df)
    y = df[target_col].copy()
    X = df.drop(columns=[target_col])
    X, y = preprocess_data(pd.concat([X, y.rename("SalePrice")], axis=1))
    return X, y
