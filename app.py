import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
from pathlib import Path

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "desire_detector.keras"
TOKENIZER_PATH = ROOT / "tokenizer.pkl"
LABELS_PATH = ROOT / "labels.pkl"

st.set_page_config(page_title="AI Desire Detector", page_icon="🧠", layout="centered")
st.title("🧠 AI Desire Detector")
st.caption("A student deep-learning demo that classifies the preference or goal expressed in text.")
st.info("This demo identifies patterns in what a person writes. It cannot read minds or reliably determine hidden desires.")

if not MODEL_PATH.exists() or not TOKENIZER_PATH.exists() or not LABELS_PATH.exists():
    st.warning("The model files are not present yet. Train the model first by running: python train_model.py")
    st.stop()

model = load_model(MODEL_PATH)
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)
with open(LABELS_PATH, "rb") as f:
    labels = pickle.load(f)

text = st.text_area(
    "Enter a sentence about a goal, preference, or interest:",
    placeholder="Example: I want to learn coding and build useful apps.",
    height=120
)
if st.button("Detect expressed preference", type="primary"):
    if not text.strip():
        st.error("Please enter a sentence first.")
    else:
        seq = tokenizer.texts_to_sequences([text.lower()])
        padded = pad_sequences(seq, maxlen=30, padding="post", truncating="post")
        probs = model.predict(padded, verbose=0)[0]
        idx = int(np.argmax(probs))
        st.success(f"Predicted category: **{labels[idx]}**")
        st.write(f"Model confidence: **{probs[idx]*100:.1f}%**")
        st.caption("Confidence is the model's score, not a guarantee that the prediction is correct.")
        st.subheader("Scores for all categories")
        for label, prob in sorted(zip(labels, probs), key=lambda x: x[1], reverse=True):
            st.write(f"**{label}** — {prob*100:.1f}%")
            st.progress(float(prob))
