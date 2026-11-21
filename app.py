
import json
import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from data_utils import CLASS_NAMES_PATH, IMG_SIZE, MODEL_PATH

st.set_page_config(page_title="RecycleVision", page_icon="♻️")


@st.cache_resource          # load the model only once, not on every click
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH) as f:
        class_names = json.load(f)
    return model, class_names


def predict(model, image):
    """Prepare the image exactly like in training and return the 6 class probabilities."""
    image = np.array(image.convert("RGB"))                # PIL image -> RGB pixel array
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))  # resize to 224x224 (same as training)
    batch = tf.expand_dims(image, axis=0)                 # model expects a batch: (1, 224, 224, 3)
    probabilities = model.predict(batch, verbose=0)[0]    # softmax output, sums to 1
    return probabilities


st.title("♻️ RecycleVision")
st.write("Garbage image classification using deep learning (MobileNetV2 transfer learning).")
st.write("Categories: **cardboard, glass, metal, paper, plastic, trash**")

if not os.path.exists(MODEL_PATH):
    st.error("Trained model not found. Run `python train.py` first.")
    st.stop()

model, class_names = load_model()

uploaded_file = st.file_uploader("Upload a garbage image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", width=300)

    probabilities = predict(model, image)
    best = int(np.argmax(probabilities))                  # index of the highest probability
    confidence = probabilities[best] * 100                # probability -> percentage

    st.success(f"Predicted category: **{class_names[best].upper()}**")
    st.metric("Confidence", f"{confidence:.1f}%")

    # Top-3 predictions
    st.subheader("Top 3 predictions")
    top3 = np.argsort(probabilities)[::-1][:3]            # indexes sorted from highest to lowest
    for i in top3:
        st.write(f"{class_names[i]}: {probabilities[i] * 100:.1f}%")
        st.progress(float(probabilities[i]))
