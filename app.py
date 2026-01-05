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
        /* Remove Streamlit header & footer */
        header {{visibility: hidden;}}
        footer {{visibility: hidden;}}

        /* Background image with dark overlay */
        .stApp {{
            background-image:
                linear-gradient(rgba(0,0,0,0.65), rgba(0,0,0,0.65)),
                url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
        }}

        /* Main content box */
        .content-box {{
            background-color: rgba(255, 255, 255, 0.93);
            padding: 28px;
            border-radius: 14px;
            margin-top: 6px;
        }}

        /* Title spacing FIX */
        h1 {{
            font-size: 44px !important;
            font-weight: 800;
            margin-bottom: 6px !important;
        }}

        h2 {{
            font-size: 28px !important;
            font-weight: 700;
            margin-top: 0px !important;
            margin-bottom: 10px !important;
        }}

        /* Reduce gap before sections */
        h3 {{
            margin-top: 8px !important;
            margin-bottom: 10px !important;
        }}

        /* Feature label text */
        label {{
            font-size: 18px !important;
            font-weight: 600;
        }}

        /* Input box size */
        div[data-baseweb="input"] input {{
            height: 46px !important;
            font-size: 16px !important;
            padding: 6px 10px !important;
        }}

        /* Reduce top padding of page */
        .block-container {{
            padding-top: 0.5rem !important;
        }}

        /* Remove extra divider spacing */
        hr {{
            margin: 8px 0px !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set background image
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

# SMALL divider (space reduced)
st.markdown("<hr>", unsafe_allow_html=True)

# ================================
# Input Section
# ================================
st.markdown("### 🔢 Enter Medical Feature Values")

input_data = []
cols = st.columns(6)   # 🔥 6 FEATURES PER ROW

for i, feature in enumerate(feature_names):
    with cols[i % 6]:
        value = st.number_input(
            label=feature,
            min_value=0.0,
            format="%.5f"
        )
        input_data.append(value)

input_array = np.array(input_data).reshape(1, -1)

st.markdown("<hr>", unsafe_allow_html=True)

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
