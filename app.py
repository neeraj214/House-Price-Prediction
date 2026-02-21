import os

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")


st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #111827 0%, #1f2933 40%, #0f172a 100%);
        color: #f9fafb;
    }
    [data-testid="stHeader"] {
        background-color: rgba(0, 0, 0, 0);
    }
    .main-title {
        text-align: center;
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #f9fafb;
    }
    .subtitle {
        text-align: center;
        font-size: 0.95rem;
        color: #9ca3af;
        margin-bottom: 1.5rem;
    }
    .card {
        background-color: rgba(17, 24, 39, 0.9);
        padding: 1.5rem 1.25rem;
        border-radius: 1rem;
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(55, 65, 81, 0.7);
    }
    .prediction-card {
        text-align: center;
    }
    .primary-button button {
        background: linear-gradient(135deg, #22c55e, #16a34a);
        color: white;
        border-radius: 999px;
        border: none;
        font-weight: 600;
        width: 100%;
    }
    .primary-button button:hover {
        background: linear-gradient(135deg, #16a34a, #15803d);
    }
    .app-footer {
        text-align: center;
        margin-top: 2rem;
        color: #9ca3af;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_pipeline(path: str = "models/best_pipeline.pkl"):
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def format_inr(value: float) -> str:
    integer_value = int(round(value))
    negative = integer_value < 0
    integer_value = abs(integer_value)
    s = str(integer_value)
    if len(s) <= 3:
        grouped = s
    else:
        last_three = s[-3:]
        remaining = s[:-3]
        parts = []
        while len(remaining) > 0:
            parts.insert(0, remaining[-2:])
            remaining = remaining[:-2]
        grouped = ",".join(parts + [last_three])
    if negative:
        grouped = "-" + grouped
    return f"₹ {grouped}"


def get_feature_importance_df(pipeline, reference_df: pd.DataFrame | None):
    estimator = pipeline
    if hasattr(estimator, "named_steps"):
        steps = list(estimator.named_steps.values())
        if steps:
            estimator = steps[-1]
    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
        if hasattr(estimator, "feature_names_in_"):
            names = list(estimator.feature_names_in_)
        elif reference_df is not None:
            names = list(reference_df.columns)
        else:
            names = [f"Feature {i}" for i in range(len(importances))]
        data = pd.DataFrame({"Feature": names[: len(importances)], "Importance": importances})
        data = data.sort_values("Importance", ascending=False).head(8)
        return data
    return None


def main():
    pipeline = load_pipeline()
    st.markdown(
        "<div class='main-title'>AI-Powered Real Estate Price Predictor</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='subtitle'>Estimate property prices instantly using a production-ready machine learning pipeline.</div>",
        unsafe_allow_html=True,
    )
    if pipeline is None:
        st.error("Trained pipeline not found at models/best_pipeline.pkl. Train and save the pipeline before using the app.")
        st.markdown(
            "<div class='app-footer'>Developed using Scikit-learn &amp; Streamlit</div>",
            unsafe_allow_html=True,
        )
        return
    left_col, right_col = st.columns([1.1, 1])
    with left_col:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Property Details")
        area = st.number_input("Area (sq ft)", min_value=200.0, max_value=20000.0, value=1200.0, step=50.0)
        bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3, step=1)
        bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2, step=1)
        age = st.number_input("Age of House (years)", min_value=0.0, max_value=100.0, value=5.0, step=1.0)
        quality = st.slider("Overall Quality (1-10)", min_value=1, max_value=10, value=7)
        st.markdown("<div class='primary-button'>", unsafe_allow_html=True)
        predict_clicked = st.button("Predict Price")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right_col:
        st.markdown("<div class='card prediction-card'>", unsafe_allow_html=True)
        st.subheader("Prediction Overview")
        price_placeholder = st.empty()
        confidence_placeholder = st.empty()
        chart_placeholder = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)
    if predict_clicked:
        input_df = pd.DataFrame(
            {
                "Area": [area],
                "Bedrooms": [bedrooms],
                "Bathrooms": [bathrooms],
                "Age": [age],
                "OverallQual": [quality],
            }
        )
        with st.spinner("Predicting house price..."):
            y_pred = pipeline.predict(input_df)
        if hasattr(y_pred, "__len__") and len(y_pred) > 0:
            predicted_value = float(y_pred[0])
        else:
            predicted_value = float(y_pred)
        formatted_price = format_inr(predicted_value)
        price_placeholder.metric(label="Predicted Price", value=formatted_price)
        confidence_placeholder.metric(label="Model Confidence", value="N/A")
        importance_df = get_feature_importance_df(pipeline, input_df)
        if importance_df is not None:
            chart_placeholder.bar_chart(importance_df.set_index("Feature"))
        st.success("Prediction generated successfully.")
    st.markdown(
        "<div class='app-footer'>Developed using Scikit-learn &amp; Streamlit</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
