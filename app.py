import joblib
from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Project paths
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

# Page configuration
st.set_page_config(
    page_title="Chennai Temperature Predictor",
    page_icon="🌡️",
    layout="wide"
)

# Dashboard heading
st.title("Chennai Temperature Prediction System")
st.write(
    "A machine-learning project using historical NASA POWER "
    "weather data to predict temperature."
)



# Model performance
st.header("Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric("XGBoost R²", "0.945")
col2.metric("XGBoost MAE", "0.409 °C")
col3.metric("7-day MAE", "0.745 °C")

st.caption(
    "Scores are from the existing model run on the historical test period."
)

# Charts
st.header("Model Visualizations")

chart_files = {
    "Actual vs Predicted Temperature": "actual_vs_predicted.png",
    "Feature Importance": "feature_importance.png",
    "Seasonal Temperature Pattern": "seasonal_pattern.png",
    "Prediction Horizon Comparison": "horizon_comparison.png",
}

for title, filename in chart_files.items():
    file_path = RESULTS_DIR / filename

    if file_path.exists():
        st.subheader(title)
        st.image(str(file_path), use_container_width=True)
    else:
        st.warning(f"Chart not found: {filename}")

# Historical data preview
st.header("Historical Dataset")

csv_path = RESULTS_DIR / "cleaned_data.csv"

if csv_path.exists():
    data = pd.read_csv(csv_path)
    st.write(f"Dataset rows: {len(data):,}")
    st.dataframe(data.tail(10), use_container_width=True)
else:
    st.warning("Cleaned dataset not found.")

st.divider()
st.caption("Intelligent Systems — Temperature Prediction Project")
# -------------------------------
# Temperature Prediction
# -------------------------------

st.header("Predict Tomorrow's Temperature")

model_path = Path("results/compact_model.pkl")
features_path = Path("results/compact_features.pkl")

if model_path.exists() and features_path.exists():
    model = joblib.load(model_path)
    features = joblib.load(features_path)

    with st.form("prediction_form"):
        today = st.number_input(
            "Today's Temperature (°C)",
            min_value=-10.0,
            max_value=60.0,
            value=29.5
        )

        avg_3day = st.number_input(
            "3-Day Average Temperature (°C)",
            min_value=-10.0,
            max_value=60.0,
            value=29.2
        )

        avg_7day = st.number_input(
            "7-Day Average Temperature (°C)",
            min_value=-10.0,
            max_value=60.0,
            value=28.8
        )

        month = st.selectbox(
            "Month",
            options=list(range(1, 13)),
            index=4,
            format_func=lambda m: [
                "January", "February", "March", "April",
                "May", "June", "July", "August",
                "September", "October", "November", "December"
            ][m - 1]
        )

        humidity = st.slider(
            "Relative Humidity (%)",
            min_value=0,
            max_value=100,
            value=70
        )

        submitted = st.form_submit_button("Predict Temperature")

    if submitted:
        inputs = pd.DataFrame(
            [[today, avg_3day, avg_7day, month, humidity]],
            columns=features
        )

        prediction = model.predict(inputs)[0]

        st.success(
            f"Predicted next-day temperature: {prediction:.2f} °C"
        )
else:
    st.error(
        "Saved model files not found. Run temperature_model.py first."
    )
    