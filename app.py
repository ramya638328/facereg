import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# -----------------------------
# Load model and labels
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("face_recognition_model.h5")

model = load_model()

with open("class_labels.json", "r") as f:
    class_indices = json.load(f)

labels = list(class_indices.keys())

IMG_SIZE = 128

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("👤 Face Recognition System")
st.write("Upload a face image to recognize the person")

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# Prediction
# -----------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=250)

    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    st.success(f"✅ Recognized Person: **{labels[class_index]}**")
    st.info(f"Confidence: **{confidence:.2f}%**")
