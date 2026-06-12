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

# ---------------- LOAD MODEL ----------------

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

st.markdown("""
### Medical Specialty Prediction

Paste a medical report and the AI model will predict the most likely medical specialty.
""")

report = st.text_area(
    "Paste Medical Report",
    height=250,
    placeholder="Enter patient medical report..."
)

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

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Predicted Specialty",
                predicted_specialty
            )

        with col2:
            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

        st.success("Prediction completed successfully.")

        # Probability chart
        try:
            labels = encoder.classes_

            probs = {
                labels[i]: float(prediction[0][i] * 100)
                for i in range(len(labels))
            }

            st.subheader("Prediction Probabilities")
            st.bar_chart(probs)

        except:
            pass

st.markdown("---")
st.caption("Healthcare NLP Classification using Deep Learning")
