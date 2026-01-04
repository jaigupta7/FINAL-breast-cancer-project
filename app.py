import streamlit as st
import joblib
import numpy as np
import os
import base64

# ================================
# Background + UI Styling
# ================================
def set_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image:
                linear-gradient(rgba(0,0,0,0.65), rgba(0,0,0,0.65)),
                url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
        }}

        .content-box {{
            background-color: rgba(255, 255, 255, 0.92);
            padding: 30px;
            border-radius: 12px;
            margin-top: 20px;
        }}

        h1 {{ font-size: 42px !important; }}
        h2 {{ font-size: 28px !important; }}

        label {{
            font-size: 16px !important;
            font-weight: 600;
        }}

        div[data-baseweb="input"] input {{
            height: 36px !important;
            padding: 4px 8px !important;
            font-size: 14px !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg("background.jpg")

# ================================
# Load ML files
# ================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "breast_cancer_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
feature_names = joblib.load(os.path.join(BASE_DIR, "feature_names.pkl"))

# ================================
# App Layout
# ================================
st.set_page_config(page_title="Cancer Detection System", layout="wide")

st.markdown('<div class="content-box">', unsafe_allow_html=True)

st.title("🧬 Cancer Detection System")
st.subheader("Machine Learning Based Cancer Detection")

st.markdown("---")

# ================================
# Input Section (BOTTOM)
# ================================
st.markdown("## 🔢 Enter Medical Feature Values")

input_data = []
cols = st.columns(5)

for i, feature in enumerate(feature_names):
    with cols[i % 5]:
        value = st.number_input(
            label=feature,
            min_value=0.0,
            format="%.5f"
        )
        input_data.append(value)

input_array = np.array(input_data).reshape(1, -1)

st.markdown("---")

# ================================
# Prediction
# ================================
if st.button("🔍 Detect Cancer"):
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]

    st.markdown("## 🧾 Prediction and Classification Results")

    if prediction == 1:
        st.error("🚨 **Cancer Detected (Malignant)**")
    else:
        st.success("✅ **No Cancer Detected (Benign)**")

    st.markdown("### 📊 Detection Probability")
    st.write(f"Benign: {probabilities[0] * 100:.2f}%")
    st.write(f"Malignant: {probabilities[1] * 100:.2f}%")

st.markdown('</div>', unsafe_allow_html=True)
