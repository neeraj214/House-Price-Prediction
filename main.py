 import pandas as pd
 import numpy as np
 import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
 import matplotlib.pyplot as plt
 import seaborn as sns
 import streamlit as st
 import joblib
import os
import json
 
 
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


def create_train_test(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_models(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    X_train, X_test, y_train, y_test = create_train_test(X, y, test_size=test_size, random_state=random_state)
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    dt_model = DecisionTreeRegressor(random_state=random_state)
    dt_model.fit(X_train, y_train)
    rf_model = RandomForestRegressor(random_state=random_state)
    rf_model.fit(X_train, y_train)
    return lr_model, dt_model, rf_model, X_train, X_test, y_train, y_test


def evaluate_models(lr_model, dt_model, rf_model, X_test: pd.DataFrame, y_test: pd.Series):
    preds = {
        "LinearRegression": lr_model.predict(X_test),
        "DecisionTreeRegressor": dt_model.predict(X_test),
        "RandomForestRegressor": rf_model.predict(X_test),
    }
    rows = []
    for name, y_pred in preds.items():
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        rows.append({"model": name, "r2": r2, "mae": mae, "rmse": rmse})
    results = pd.DataFrame(rows, columns=["model", "r2", "mae", "rmse"])
    print("Model | R2 | MAE | RMSE")
    for _, row in results.iterrows():
        print(f"{row['model']} | {row['r2']:.4f} | {row['mae']:.2f} | {row['rmse']:.2f}")
    return results


def select_best_by_r2(lr_model, dt_model, rf_model, X_test: pd.DataFrame, y_test: pd.Series):
    r2_scores = {
        "LinearRegression": r2_score(y_test, lr_model.predict(X_test)),
        "DecisionTreeRegressor": r2_score(y_test, dt_model.predict(X_test)),
        "RandomForestRegressor": r2_score(y_test, rf_model.predict(X_test)),
    }
    best_name = max(r2_scores.items(), key=lambda kv: kv[1])[0]
    models = {
        "LinearRegression": lr_model,
        "DecisionTreeRegressor": dt_model,
        "RandomForestRegressor": rf_model,
    }
    return best_name, models[best_name]


def plot_rf_feature_importance(rf_model: RandomForestRegressor, feature_names, top_n: int = 20):
    importances = rf_model.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]
    names = [feature_names[i] for i in idx]
    vals = importances[idx]
    plt.figure(figsize=(10, max(4, int(0.4 * len(names)))))
    plt.barh(range(len(names))[::-1], vals[::-1])
    plt.yticks(range(len(names))[::-1], names[::-1])
    plt.xlabel("Importance")
    plt.title("RandomForest Feature Importances")
    plt.tight_layout()
    plt.show()


def select_and_save_best(lr_model, dt_model, rf_model, X_test: pd.DataFrame, y_test: pd.Series, model_dir: str = "models"):
    results = evaluate_models(lr_model, dt_model, rf_model, X_test, y_test)
    models = {
        "LinearRegression": lr_model,
        "DecisionTreeRegressor": dt_model,
        "RandomForestRegressor": rf_model,
    }
    best_idx = results["rmse"].idxmin()
    best_name = results.loc[best_idx, "model"]
    best_model = models[best_name]
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(best_model, os.path.join(model_dir, "best_model.joblib"))
    with open(os.path.join(model_dir, "feature_columns.json"), "w", encoding="utf-8") as f:
        json.dump(list(X_test.columns), f)
    with open(os.path.join(model_dir, "evaluation_results.json"), "w", encoding="utf-8") as f:
        json.dump(results.to_dict(orient="records"), f)
    return best_name, best_model, results
