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
 
 
 def load_data(path: str = "data/raw/train.csv"):
     df = pd.read_csv(path)
     X, y = preprocess_data(df)
     return X, y
