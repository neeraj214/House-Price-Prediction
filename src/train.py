import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline


def create_train_test(X: pd.DataFrame, y, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_pipeline(X: pd.DataFrame, y, preprocessor, test_size: float = 0.2, random_state: int = 42):
    X_train, X_test, y_train, y_test = create_train_test(X, y, test_size=test_size, random_state=random_state)
    model = RandomForestRegressor(random_state=random_state)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    pipeline.fit(X_train, y_train)
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, os.path.join("models", "pipeline.pkl"))
    return pipeline, X_train, X_test, y_train, y_test
