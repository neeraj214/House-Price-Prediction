import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def preprocess_data(df: pd.DataFrame):
    print(f"Dataset shape: {df.shape}")
    print(f"Total missing values: {int(df.isnull().sum().sum())}")
    y = df["SalePrice"].copy()
    X = df.drop(columns=["SalePrice"])
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )
    return X, y, preprocessor


def load_data(path: str = "data/raw/train.csv"):
    df = pd.read_csv(path)
    X, y, preprocessor = preprocess_data(df)
    return X, y, preprocessor
