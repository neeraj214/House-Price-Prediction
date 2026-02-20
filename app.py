import streamlit as st
import pandas as pd
import joblib
import json
import os
 
 
def load_artifacts(stem: str, model_dir="models"):
    model_path = os.path.join(model_dir, f"model_{stem}.pkl")
    cols_path = os.path.join(model_dir, f"feature_columns_{stem}.json")
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


def list_model_stems(model_dir="models"):
    stems = []
    for fname in os.listdir(model_dir):
        if fname.startswith("model_") and fname.endswith(".pkl"):
            stem = fname[len("model_"):-len(".pkl")]
            stems.append(stem)
    return sorted(stems)
 
 
def main():
    st.title("House Price Prediction")
    stems = list_model_stems()
    if not stems:
        st.warning("No saved models found in models/. Train first to create model files.")
        return
    stem = st.selectbox("Select saved model", stems, index=0)
    st.write("Upload a CSV with preprocessed features matching training columns.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        model, feature_columns = load_artifacts(stem)
        df_aligned = align_columns(df, feature_columns)
        preds = model.predict(df_aligned)
        out = pd.DataFrame({"Prediction": preds})
        st.write(out)
 
 
if __name__ == "__main__":
     main()
