import streamlit as st
import numpy as np
import tensorflow as tf
import pickle

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Healthcare NLP Dashboard",
    page_icon="🏥",
    layout="wide"
)

# ---------------- LOAD FILES ----------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.keras")

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
    st.error(f"Error loading files: {e}")
    st.stop()

# ---------------- SETTINGS ----------------

MAX_LEN = 250

# ---------------- UI ----------------

st.title("🏥 Healthcare NLP Dashboard")
st.markdown(
    "Predict the **Medical Specialty** from a clinical report using Deep Learning."
)

report = st.text_area(
    "Paste Medical Report",
    height=250,
    placeholder="Enter patient report here..."
)

if st.button("Predict Specialty"):

    if not report.strip():
        st.warning("Please enter a medical report.")
    else:

        seq = tokenizer.texts_to_sequences([report])

        pad = tf.keras.preprocessing.sequence.pad_sequences(
            seq,
            maxlen=MAX_LEN,
            padding="post"
        )

        pred = model.predict(pad, verbose=0)

        index = np.argmax(pred)

        specialty = encoder.inverse_transform([index])[0]

        confidence = float(np.max(pred) * 100)

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

        st.success("Prediction completed successfully.")
