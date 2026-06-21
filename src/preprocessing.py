
import os
import glob
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def clean_size(size_str):
    """Clean size string to get numeric value in sqft."""
    if not isinstance(size_str, str):
        return float(size_str)
    # Handle ranges like "1085-1300 sqft"
    if "-" in size_str:
        parts = size_str.split("-")
        nums = []
        for part in parts:
            match = re.search(r'(\d+(\.\d+)?)', part)
            if match:
                nums.append(float(match.group(1)))
        if nums:
            return np.mean(nums)
    # Handle single value like "1200 sqft"
    match = re.search(r'(\d+(\.\d+)?)', size_str)
    if match:
        return float(match.group(1))
    return 0.0


def preprocess_data(df: pd.DataFrame):
    print(f"Dataset shape: {df.shape}")
    print(f"Total missing values: {int(df.isnull().sum().sum())}")
    
    # Clean and prepare data
    df = df.copy()
    
    # Drop irrelevant columns
    if 'url' in df.columns:
        df = df.drop(columns=['url'])
    if 'date' in df.columns:
        df = df.drop(columns=['date'])
    
    # Clean size column
    if 'size' in df.columns:
        df['size_sqft'] = df['size'].apply(clean_size)
        df = df.drop(columns=['size'])
    
    # Separate target
    y = df["SalePrice"].copy()
    X = df.drop(columns=["SalePrice"])
    
    # Filter out rows where price is 0 or size_sqft is 0
    valid_indices = (y > 0) & (X['size_sqft'] > 0)
    X = X[valid_indices]
    y = y[valid_indices]
    
    print(f"Filtered dataset shape: {X.shape}")
    
    # Split into numeric and categorical
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    
    # Handle numeric columns
    if numeric_cols:
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
    
    # Handle categorical columns
    for col in categorical_cols:
        mode_vals = X[col].mode(dropna=True)
        fill_val = mode_vals.iloc[0] if len(mode_vals) else "Unknown"
        X[col] = X[col].fillna(fill_val)
    
    # One-hot encode categorical columns
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Scale numeric columns
    if numeric_cols:
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    return X, y


def _auto_find_dataset():
    candidates = [
        "dataset_for_house_price/real_estate_dataset.csv",
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
    for pattern in ["**/real_estate_dataset.csv", "**/data(1).csv", "**/Housing(1).csv", "**/train.csv"]:
        found = glob.glob(pattern, recursive=True)
        if found:
            return found[0]
    raise FileNotFoundError("Dataset not found.")


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
