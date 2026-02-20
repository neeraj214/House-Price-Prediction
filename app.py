import streamlit as st
import pandas as pd
import joblib
import json
import os
 
 
def load_artifacts(model_dir="models"):
     model_path = os.path.join(model_dir, "model.pkl")
     cols_path = os.path.join(model_dir, "feature_columns.json")
     model = joblib.load(model_path)
     with open(cols_path, "r", encoding="utf-8") as f:
         feature_columns = json.load(f)
     return model, feature_columns
 
 
def align_columns(df: pd.DataFrame, feature_columns):
     for col in feature_columns:
         if col not in df.columns:
             df[col] = 0
     df = df[feature_columns]
     return df
 
 
def main():
     st.title("House Price Prediction")
     st.write("Upload a CSV with preprocessed features matching training columns.")
     uploaded = st.file_uploader("Upload CSV", type=["csv"])
     if uploaded is not None:
         df = pd.read_csv(uploaded)
         model, feature_columns = load_artifacts()
         df_aligned = align_columns(df, feature_columns)
         preds = model.predict(df_aligned)
         out = pd.DataFrame({"Prediction": preds})
         st.write(out)
 
 
if __name__ == "__main__":
     main()
