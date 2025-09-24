import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load trained ResNet model
model = tf.keras.models.load_model("resnet_model.h5")

# Class labels
class_labels = ["Normal", "COVID-19", "Viral Pneumonia"]

# App title
st.title("COVID-19 & Pneumonia Detection")
st.write("Upload a chest X-ray image to classify it into Normal, COVID-19, or Viral Pneumonia.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess: resize + normalize
    img = image.resize((224, 224))  # Change if your model uses a different input size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)

    # Make prediction
    prediction = model.predict(img_array)[0]

    # Get class and confidence
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"Prediction: {predicted_class} ({confidence:.2f}% confidence)")

    # Show probabilities for all classes
    st.subheader("Prediction Probabilities")
    for label, prob in zip(class_labels, prediction):
        st.write(f"{label}: {prob*100:.2f}%")
