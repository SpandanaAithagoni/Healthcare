import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

model = tf.keras.models.load_model(
    "attention_model.keras"
)

tokenizer = joblib.load(
    "tokenizer.pkl"
)

encoder = joblib.load(
    "label_encoder.pkl"
)

MAX_LEN = 250

st.title(
    "Healthcare NLP Dashboard"
)

report = st.text_area(
    "Paste Medical Report"
)

if report:

    seq = tokenizer.texts_to_sequences(
        [report]
    )

    pad = tf.keras.preprocessing.sequence.pad_sequences(
        seq,
        maxlen=MAX_LEN,
        padding="post"
    )

    pred = model.predict(
        pad
    )

    index = np.argmax(pred)

    specialty = encoder.inverse_transform(
        [index]
    )[0]

    confidence = np.max(pred)

    st.subheader(
        "Predicted Specialty"
    )

    st.write(
        specialty
    )

    st.subheader(
        "Confidence Score"
    )

    st.write(
        float(confidence)
    )