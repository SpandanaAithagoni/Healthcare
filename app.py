import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import os

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Healthcare NLP Dashboard",
    page_icon="🏥",
    layout="wide"
)

# ---------------- DEBUG SECTION ----------------

st.sidebar.header("System Diagnostics")

st.sidebar.write("Current Directory:")
st.sidebar.code(os.getcwd())

st.sidebar.write("Available Files:")
st.sidebar.write(os.listdir("."))

# ---------------- FILE CHECKS ----------------

required_files = [
    "attention_model.keras",
    "tokenizer.pkl",
    "label_encoder.pkl"
]

missing_files = []

for file in required_files:
    if not os.path.exists(file):
        missing_files.append(file)

if missing_files:
    st.error(
        f"Missing files: {', '.join(missing_files)}"
    )
    st.stop()

# ---------------- LOAD RESOURCES ----------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "attention_model.keras"
    )

@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_encoder():
    with open("label_encoder.pkl", "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
    tokenizer = load_tokenizer()
    encoder = load_encoder()

except Exception as e:
    st.error(f"Loading Error: {e}")
    st.stop()

# ---------------- SETTINGS ----------------

MAX_LEN = 250

# ---------------- HEADER ----------------

st.title("🏥 Healthcare NLP Dashboard")

st.markdown("""
### Medical Specialty Prediction System

Paste a medical report and predict the most likely specialty.
""")

# ---------------- INPUT ----------------

report = st.text_area(
    "Paste Medical Report",
    height=250
)

# ---------------- PREDICTION ----------------

if st.button("Predict Specialty"):

    if not report.strip():
        st.warning("Please enter a medical report.")
    else:

        seq = tokenizer.texts_to_sequences(
            [report]
        )

        padded = tf.keras.preprocessing.sequence.pad_sequences(
            seq,
            maxlen=MAX_LEN,
            padding="post",
            truncating="post"
        )

        prediction = model.predict(
            padded,
            verbose=0
        )

        predicted_index = np.argmax(
            prediction
        )

        specialty = encoder.inverse_transform(
            [predicted_index]
        )[0]

        confidence = float(
            np.max(prediction) * 100
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Predicted Specialty",
                specialty
            )

        with col2:
            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

        st.success(
            "Prediction completed successfully."
        )

# ---------------- FOOTER ----------------

st.markdown("---")
st.caption(
    "Healthcare NLP Classification using Deep Learning"
)
