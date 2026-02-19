import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
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
     rf_model = RandomForestRegressor(random_state=random_state)
     rf_model.fit(X_train, y_train)
     return lr_model, dt_model, rf_model, X_train, X_test, y_train, y_test
 
 
 def save_model(model, feature_columns, model_dir: str = "models", filename: str = "model.pkl"):
     os.makedirs(model_dir, exist_ok=True)
     joblib.dump(model, os.path.join(model_dir, filename))
     with open(os.path.join(model_dir, "feature_columns.json"), "w", encoding="utf-8") as f:
         json.dump(list(feature_columns), f)
