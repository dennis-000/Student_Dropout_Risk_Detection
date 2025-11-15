import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -------------------------
# 1. CONFIG & MODEL LOADING
# -------------------------

st.set_page_config(
    page_title="Student Dropout Risk Prediction",
    page_icon="🎓",
    layout="centered",
)

st.title("🎓 Student Dropout Risk Prediction")
st.write(
    "This tool uses a trained machine learning model to estimate the probability "
    "that a student will **drop out** based on their academic and background data."
)


MODEL_PATH = os.path.join("models", "best_kaggle_XGBoost.joblib")

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at `{MODEL_PATH}`. Please check the path and file name.")
    st.stop()

# Load trained pipeline (preprocessor + model)
pipeline = joblib.load(MODEL_PATH)

# Load dataset to get feature names and typical ranges
DATASET_PATH = "dataset.csv"
if not os.path.exists(DATASET_PATH):
    st.error(f"Dataset file `dataset.csv` not found in project folder.")
    st.stop()

df = pd.read_csv(DATASET_PATH)

if "Target" not in df.columns:
    st.error("Expected a 'Target' column in dataset.csv but did not find one.")
    st.stop()

feature_cols = [c for c in df.columns if c != "Target"]
X = df[feature_cols]

# -------------------------
# 2. SIDEBAR INPUT FORM
# -------------------------

st.sidebar.header("📥 Input Student Features")

st.sidebar.write(
    "Provide the values for each feature below. "
    "You can start with the defaults (based on dataset medians) and adjust."
)

# Precompute medians for numeric defaults
medians = X.median(numeric_only=True)

user_input = {}

for col in feature_cols:
    col_data = X[col]

    # If column is numeric, use number_input
    if np.issubdtype(col_data.dtype, np.number):
        col_min = float(col_data.min())
        col_max = float(col_data.max())
        default = float(medians.get(col, col_min))

        user_input[col] = st.sidebar.number_input(
            label=col,
            min_value=col_min,
            max_value=col_max,
            value=default,
            step=1.0 if col_data.dtype in [np.int64, np.int32] else 0.1,
        )
    else:
        # For any non-numeric columns (if they exist), use selectbox
        unique_vals = sorted(col_data.dropna().unique().tolist())
        if len(unique_vals) == 0:
            unique_vals = ["Unknown"]
        default = unique_vals[0]
        user_input[col] = st.sidebar.selectbox(col, unique_vals, index=0)

# Convert user_input to DataFrame with one row
input_df = pd.DataFrame([user_input])

st.subheader("📊 Input Summary")
st.write("These are the values that will be fed into the model:")
st.dataframe(input_df)

# -------------------------
# 3. PREDICTION
# -------------------------

if st.button("🔮 Predict Dropout Risk"):
    try:
        # Predict probability of class 1 = Dropout
        proba = pipeline.predict_proba(input_df)[:, 1][0]
        pred_class = pipeline.predict(input_df)[0]

        dropout_prob = float(proba)
        not_dropout_prob = 1 - dropout_prob

        # Risk band
        if dropout_prob >= 0.7:
            risk_level = "High Risk"
            color = "red"
        elif dropout_prob >= 0.4:
            risk_level = "Medium Risk"
            color = "orange"
        else:
            risk_level = "Low Risk"
            color = "green"

        st.markdown("---")
        st.subheader("🎯 Prediction Result")

        st.markdown(
            f"**Predicted Dropout Probability:** "
            f"<span style='font-size:24px; color:{color};'> {dropout_prob:.2%}</span>",
            unsafe_allow_html=True,
        )

        st.write(f"**Risk Category:** {risk_level}")

        st.write(f"**Model prediction (class):** {'Dropout (1)' if pred_class == 1 else 'Not Dropout (0)'}")

        st.markdown("#### Probability Breakdown")
        st.write(f"- Not Dropout (0): **{not_dropout_prob:.2%}**")
        st.write(f"- Dropout (1): **{dropout_prob:.2%}**")

        st.info(
            "Note: This prediction is based on historical data patterns in the Kaggle "
            "student dataset and should be used as a decision-support tool, not as a final decision."
        )

    except Exception as e:
        st.error(f"An error occurred while making prediction: {e}")
