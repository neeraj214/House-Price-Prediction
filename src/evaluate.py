import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


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
