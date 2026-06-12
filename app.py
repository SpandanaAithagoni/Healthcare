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

# ---------------- LOAD RESOURCES ----------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("attention_model.keras")

@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_encoder():
    with open("label_encoder.pkl", "rb") as f:
        return pickle.load(f)

# ---------------- INITIALIZATION ----------------

try:
    model = load_model()
    tokenizer = load_tokenizer()
    encoder = load_encoder()

except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# ---------------- SETTINGS ----------------

MAX_LEN = 250

# ---------------- HEADER ----------------

st.title("🏥 Healthcare NLP Dashboard")

st.markdown("""
### Medical Specialty Prediction System

Paste a clinical report below and the model will predict the most likely medical specialty.
""")

# ---------------- INPUT ----------------

report = st.text_area(
    "Paste Medical Report",
    height=250,
    placeholder="Enter clinical report here..."
)

# ---------------- PREDICTION ----------------

if st.button("Predict Specialty"):

    if not report.strip():
        st.warning("Please enter a medical report.")
    else:

        seq = tokenizer.texts_to_sequences([report])

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

        predicted_index = np.argmax(prediction)

        predicted_specialty = encoder.inverse_transform(
            [predicted_index]
        )[0]

        confidence = float(
            np.max(prediction) * 100
        )

        st.success(
            f"Predicted Specialty: {predicted_specialty}"
        )

        st.info(
            f"Confidence Score: {confidence:.2f}%"
        )

        st.subheader("Prediction Probabilities")

        probabilities = prediction[0]

        try:
            labels = encoder.classes_

            prob_dict = {
                labels[i]: float(probabilities[i] * 100)
                for i in range(len(labels))
            }

            prob_dict = dict(
                sorted(
                    prob_dict.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
            )

            st.bar_chart(prob_dict)

        except Exception:
            pass

# ---------------- FOOTER ----------------

st.markdown("---")
st.caption("Healthcare NLP Classification using Deep Learning")
