import streamlit as st
import joblib
import numpy as np
import os

# ================================
# Load files using absolute paths
# ================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "breast_cancer_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
feature_names = joblib.load(os.path.join(BASE_DIR, "feature_names.pkl"))

# ================================
# Streamlit UI
# ================================
st.set_page_config(
    page_title="Breast Cancer Prediction",
    layout="wide"
)

st.title("🩺 Breast Cancer Prediction System")
st.write(
    "This app predicts whether a breast tumor is **Benign** or **Malignant** "
    "using a trained Machine Learning model."
)

st.markdown("---")

# Sidebar for inputs
st.sidebar.header("Enter Tumor Features")

input_data = []

for feature in feature_names:
    value = st.sidebar.number_input(
        label=feature,
        min_value=0.0,
        format="%.5f"
    )
    input_data.append(value)

# Convert input to numpy array
input_array = np.array(input_data).reshape(1, -1)

# ================================
# Prediction
# ================================
if st.button("🔍 Predict"):
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]

    st.markdown("## 🧾 Prediction Result")

    if prediction == 1:
        st.error("🚨 **Malignant Tumor Detected**")
    else:
        st.success("✅ **Benign Tumor Detected**")

    st.markdown("### 📊 Prediction Probability")
    st.write(f"Benign: {probabilities[0] * 100:.2f}%")
    st.write(f"Malignant: {probabilities[1] * 100:.2f}%")

st.markdown("---")
st.caption("⚠️ This application is for educational purposes only and not a medical diagnosis.")
