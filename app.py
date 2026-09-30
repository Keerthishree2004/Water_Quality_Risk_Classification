from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.config import FEATURES, TARGET, RISK_CLASSES

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "water_quality_logistic_regression.joblib"
DATA_PATH = ROOT / "data" / "water_quality_dataset.csv"

st.set_page_config(
    page_title="Water Quality Risk Classification",
    page_icon="💧",
    layout="wide",
)

st.title("💧 Water Quality Risk Classification")
st.caption(
    "Academic prototype for classifying water samples into Low, Moderate and High risk."
)

if not MODEL_PATH.exists():
    st.error("Model file not found. Run `python main.py` first.")
    st.stop()

model = joblib.load(MODEL_PATH)

if "history" not in st.session_state:
    st.session_state.history = []


def predict(values):
    row = pd.DataFrame([values])
    prediction = model.predict(row)[0]
    probabilities = model.predict_proba(row)[0]
    return prediction, dict(zip(model.classes_, probabilities))


def recommendation(risk):
    recommendations = {
        "High": "Prioritize sample review and additional water-quality investigation.",
        "Moderate": "Continue monitoring and consider additional testing.",
        "Low": "Continue routine monitoring and the planned testing schedule.",
    }
    return recommendations[risk]


page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Dashboard",
        "🔮 Single Prediction",
        "📤 Batch Prediction",
        "🚨 Risk Alerts",
        "📈 Model Explanation",
        "📋 Prediction History",
        "ℹ️ About Project",
    ],
)


# ============================================================
# Dashboard
# ============================================================
if page == "📊 Dashboard":
    st.header("Water Quality Dashboard")
    data = pd.read_csv(DATA_PATH)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Samples", len(data))
    col2.metric("Low Risk", int((data[TARGET] == "Low").sum()))
    col3.metric("Moderate Risk", int((data[TARGET] == "Moderate").sum()))
    col4.metric("High Risk", int((data[TARGET] == "High").sum()))

    st.subheader("Risk Distribution")
    counts = data[TARGET].value_counts().reindex(RISK_CLASSES).fillna(0)
    st.bar_chart(counts)

    st.subheader("Average Water Properties")
    st.dataframe(
        data[FEATURES].mean(numeric_only=True).round(2).to_frame("Average"),
        use_container_width=True,
    )

    st.subheader("Dataset Preview")
    st.dataframe(data.head(10), use_container_width=True, hide_index=True)


# ============================================================
# Single prediction
# ============================================================
elif page == "🔮 Single Prediction":
    st.header("Single Water Sample Prediction")

    left, right = st.columns(2)

    with left:
        ph = st.number_input("pH", 0.0, 14.0, 7.0, 0.1)
        turbidity = st.number_input("Turbidity (NTU)", 0.0, 200.0, 5.0, 0.5)
        temperature = st.number_input("Temperature (°C)", 0.0, 60.0, 25.0, 0.5)

    with right:
        dissolved_oxygen = st.number_input(
            "Dissolved Oxygen (mg/L)", 0.0, 20.0, 7.0, 0.1
        )
        conductivity = st.number_input(
            "Conductivity (µS/cm)", 0.0, 5000.0, 700.0, 10.0
        )

    values = {
        "ph": ph,
        "turbidity_ntu": turbidity,
        "temperature_c": temperature,
        "dissolved_oxygen_mg_l": dissolved_oxygen,
        "conductivity_us_cm": conductivity,
    }

    if st.button("🔮 Predict Risk", type="primary"):
        risk, probabilities = predict(values)

        if risk == "High":
            st.error("🚨 Risk Category: High")
        elif risk == "Moderate":
            st.warning("⚠️ Risk Category: Moderate")
        else:
            st.success("✅ Risk Category: Low")

        st.subheader("Class Probabilities")
        st.bar_chart(pd.Series(probabilities, name="Probability"))
        st.dataframe(
            pd.DataFrame(
                {
                    "Risk Category": list(probabilities.keys()),
                    "Probability": [
                        f"{p * 100:.2f}%" for p in probabilities.values()
                    ],
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Suggested Next Step")
        st.info(recommendation(risk))

        record = values.copy()
        record.update(
            {
                "prediction": risk,
                "probability": probabilities[risk],
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        st.session_state.history.append(record)


# ============================================================
# Batch prediction
# ============================================================
elif page == "📤 Batch Prediction":
    st.header("Batch CSV Prediction")
    st.write("Required columns:")
    st.code(
        "ph, turbidity_ntu, temperature_c, "
        "dissolved_oxygen_mg_l, conductivity_us_cm"
    )

    uploaded = st.file_uploader("Upload water-sample CSV", type=["csv"])

    if uploaded:
        batch = pd.read_csv(uploaded)
        missing = [feature for feature in FEATURES if feature not in batch.columns]

        if missing:
            st.error("Missing columns: " + ", ".join(missing))
        else:
            result = batch.copy()
            result["predicted_risk"] = model.predict(batch[FEATURES])
            probabilities = model.predict_proba(batch[FEATURES])

            for index, label in enumerate(model.classes_):
                result[f"probability_{label.lower()}"] = probabilities[:, index]

            st.success(f"Predictions completed for {len(result)} samples.")
            st.dataframe(result, use_container_width=True)

            summary = (
                result["predicted_risk"]
                .value_counts()
                .reindex(RISK_CLASSES)
                .fillna(0)
            )
            st.subheader("Batch Risk Summary")
            st.bar_chart(summary)

            st.download_button(
                "📥 Download Prediction Results",
                result.to_csv(index=False).encode("utf-8"),
                "water_quality_predictions.csv",
                "text/csv",
            )


# ============================================================
# Risk alerts
# ============================================================
elif page == "🚨 Risk Alerts":
    st.header("Risk Alerts")
    data = pd.read_csv(DATA_PATH)
    high = data[data[TARGET] == "High"]
    moderate = data[data[TARGET] == "Moderate"]

    left, right = st.columns(2)

    with left:
        st.error(f"🚨 High Risk Samples: {len(high)}")
        st.dataframe(high.head(20), use_container_width=True, hide_index=True)

    with right:
        st.warning(f"⚠️ Moderate Risk Samples: {len(moderate)}")
        st.dataframe(moderate.head(20), use_container_width=True, hide_index=True)

    st.info(
        "For real environmental decisions, measurements should be validated "
        "against applicable standards and professional review."
    )


# ============================================================
# Model explanation
# ============================================================
elif page == "📈 Model Explanation":
    st.header("Model Explanation")
    st.write(
        "The chart shows mean absolute Logistic Regression coefficient magnitude "
        "as a model-level feature indicator; it is not causal evidence."
    )

    classifier = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    importance = np.mean(np.abs(classifier.coef_), axis=0)

    explanation = pd.DataFrame(
        {
            "Feature": feature_names,
            "Mean Absolute Coefficient": importance,
        }
    ).sort_values("Mean Absolute Coefficient", ascending=False)

    st.bar_chart(explanation.set_index("Feature"))
    st.dataframe(explanation, use_container_width=True, hide_index=True)

    st.subheader("Prediction Flow")
    st.markdown(
        """
        **Water sample → Missing-value handling → Scaling → Logistic Regression → Risk category**

        1. Enter the five capstone inputs.
        2. Missing values are median-imputed.
        3. Features are standardized.
        4. Logistic Regression calculates class probabilities.
        5. The highest-probability class is displayed.
        """
    )


# ============================================================
# Prediction history
# ============================================================
elif page == "📋 Prediction History":
    st.header("Prediction History")

    if not st.session_state.history:
        st.info("No predictions have been made in this session yet.")
    else:
        history = pd.DataFrame(st.session_state.history)
        st.dataframe(history, use_container_width=True, hide_index=True)

        st.download_button(
            "📥 Download Prediction History",
            history.to_csv(index=False).encode("utf-8"),
            "water_quality_prediction_history.csv",
            "text/csv",
        )

        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()


# ============================================================
# About
# ============================================================
else:
    st.header("About Project")
    st.markdown(
        """
        **Domain:** Environment

        **ML Task:** Classification

        **Algorithm:** Logistic Regression

        **Expected inputs:** pH, Turbidity, Temperature, Dissolved Oxygen and Conductivity.

        **Application features:** Dashboard, single prediction, batch CSV prediction,
        risk alerts, probability visualization, model explanation and prediction history.
        """
    )
    st.warning(
        "Academic prototype using synthetic data; not a certified water-safety determination."
    )
