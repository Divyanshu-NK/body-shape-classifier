import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the trained model
MODEL_PATH = 'body_shape_model.h5'
model = load_model(MODEL_PATH)

# Define class labels
CLASS_LABELS = ['Apple-shaped', 'Hourglass', 'Inverted Triangle', 'Pear-shaped', 'Rectangle']

# Function to preprocess image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Streamlit app layout
st.title("Body Shape Classification App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    file_path = "temp_image.jpg"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    # Display uploaded image
    st.image(file_path, caption="Uploaded Image", use_column_width=True)

    # Predict
    img_array = preprocess_image(file_path)
    prediction = model.predict(img_array)
    predicted_class = CLASS_LABELS[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Display result
    st.success(f"Prediction: **{predicted_class}**")
    st.info(f"Confidence: {confidence:.2f}%")
