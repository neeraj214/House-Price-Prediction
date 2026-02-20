import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
 
 
def create_train_test(X: pd.DataFrame, y, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
 
 
def train_models(X: pd.DataFrame, y, test_size: float = 0.2, random_state: int = 42):
    X_train, X_test, y_train, y_test = create_train_test(X, y, test_size=test_size, random_state=random_state)
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    dt_model = DecisionTreeRegressor(random_state=random_state)
    dt_model.fit(X_train, y_train)
    rf_base = RandomForestRegressor(random_state=random_state)
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2"],
    }
    grid = GridSearchCV(rf_base, param_grid, cv=5, scoring="r2", n_jobs=-1)
    grid.fit(X_train, y_train)
    rf_model = grid.best_estimator_
    return lr_model, dt_model, rf_model, X_train, X_test, y_train, y_test
 
 
def save_model(model, feature_columns, model_dir: str = "models", filename: str = "model.pkl", columns_filename: str = "feature_columns.json"):
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, filename))
    with open(os.path.join(model_dir, columns_filename), "w", encoding="utf-8") as f:
        json.dump(list(feature_columns), f)
