"""
COS30082 Assignment 1 - Bird Species Classifier web app
Samuel Chang Hao Jun (102775957)

Streamlit app that lets a user upload a bird photo and shows the 5 most likely
species out of the 200 CUB-200 classes.

The model is a frozen ImageNet EfficientNetB0 with a softmax regression layer on top
(approach 2 in the report). It was exported from notebook 4 as one Keras file that
takes a raw 224 x 224 RGB image (pixel values 0 to 255) and outputs 200 probabilities.
"""

import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # hide TensorFlow start-up messages

import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

# ---- Files exported by notebook 4 ----
APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "model" / "bird_classifier.keras"
CLASSES_PATH = APP_DIR / "model" / "class_names.json"
INFO_PATH = APP_DIR / "model" / "model_info.json"

IMG_SIZE = 224          # same size the model was trained on
TOP_K = 5               # how many species to show
LOW_CONFIDENCE = 0.30   # below this the app warns that the prediction is uncertain

st.set_page_config(page_title="Bird species classifier", page_icon="🐦", layout="centered")

# Small style tweaks so the top prediction stands out from the runners-up
st.markdown(
    """
    <style>
    .top-name {font-size: 2.1rem; font-weight: 700; line-height: 1.15; margin: 0.2rem 0 0.1rem 0;}
    .top-conf {font-size: 1.05rem; opacity: 0.8; margin-bottom: 1rem;}
    .runner {display: flex; justify-content: space-between; font-size: 0.98rem; margin-top: 0.55rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the model, this takes a few seconds the first time")
def load_model():
    # Loaded once and then reused for every user and every prediction
    import keras   # imported here so the page appears before TensorFlow finishes loading
    return keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_names():
    with open(CLASSES_PATH) as f:
        return json.load(f)


@st.cache_data
def load_model_info():
    if INFO_PATH.exists():
        with open(INFO_PATH) as f:
            return json.load(f)
    return {}


def preprocess(image):
    # Same steps as training. Fix phone photo rotation, force RGB, resize to 224 x 224
    # with bilinear resizing (what tf.image.resize used in the notebooks), keep 0-255 values.
    import tensorflow as tf
    image = ImageOps.exif_transpose(image).convert("RGB")
    arr = np.asarray(image, dtype=np.float32)
    arr = tf.image.resize(arr, [IMG_SIZE, IMG_SIZE]).numpy()
    arr = np.clip(arr, 0, 255)
    return arr[np.newaxis, ...]           # add the batch dimension, shape (1, 224, 224, 3)


def predict(model, image):
    probs = model.predict(preprocess(image), verbose=0)[0]
    top = np.argsort(probs)[::-1][:TOP_K]  # indices of the 5 highest probabilities
    return [(int(i), float(probs[i])) for i in top]


# ---- Sidebar, about the model ----
info = load_model_info()
with st.sidebar:
    st.header("About")
    st.write(
        "This app identifies 200 bird species, mostly North American, "
        "from the Caltech-UCSD Birds 200 dataset."
    )
    st.write(
        "It uses an EfficientNetB0 network pretrained on ImageNet as a fixed feature extractor, "
        "with a softmax regression classifier trained on the bird images."
    )
    if info:
        st.subheader("Test set results")
        st.write(f"Top-1 accuracy {info['test_top1'] * 100:.2f}%")
        st.write(f"Average accuracy per class {info['test_avg_per_class'] * 100:.2f}%")
        st.write(f"Top-5 accuracy {info['test_top5'] * 100:.2f}%")
    st.caption("COS30082 Applied Machine Learning\n\nSamuel Chang Hao Jun, 102775957")

# ---- Main page ----
st.title("Bird species classifier")
st.write("Upload a photo of a bird and the model will suggest the five most likely species.")

uploaded = st.file_uploader("Bird photo", type=["jpg", "jpeg", "png", "webp", "bmp"])

if uploaded is None:
    st.info("Choose a JPG or PNG photo to get a prediction. Photos where the bird fills most of the frame work best.")
    st.stop()

try:
    image = Image.open(uploaded)
    image.load()
except (UnidentifiedImageError, OSError):
    st.error("This file could not be opened as an image. Upload a JPG or PNG photo instead.")
    st.stop()

class_names = load_class_names()
model = load_model()

with st.spinner("Identifying the bird"):
    results = predict(model, image)

col_img, col_pred = st.columns([1, 1], gap="large")

with col_img:
    st.image(ImageOps.exif_transpose(image), width="stretch")

with col_pred:
    best_idx, best_p = results[0]
    st.markdown(f'<div class="top-name">{class_names[best_idx]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="top-conf">{best_p * 100:.2f}% confidence</div>', unsafe_allow_html=True)
    st.progress(best_p)

    st.markdown("**Other possible species**")
    for idx, p in results[1:]:
        st.markdown(
            f'<div class="runner"><span>{class_names[idx]}</span><span>{p * 100:.2f}%</span></div>',
            unsafe_allow_html=True,
        )
        st.progress(p)

if best_p < LOW_CONFIDENCE:
    st.warning(
        "The model is not confident about this photo. The bird may be small or partly hidden, "
        "or it may not be one of the 200 species the model knows."
    )

with st.expander("How the prediction is made"):
    st.write(
        "The photo is resized to 224 x 224 pixels and passed through EfficientNetB0, which turns it into "
        "1280 numbers describing shapes, colours and textures. A softmax regression layer then turns those "
        "numbers into a probability for each of the 200 species. The five highest are shown above."
    )
