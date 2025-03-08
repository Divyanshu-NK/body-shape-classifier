import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import gdown  # For downloading large models
from PIL import Image

# Download Model from Google Drive
MODEL_PATH = "body_shape_model.h5"

@st.cache_resource
def load_model():
    try:
        gdown.download('https://drive.google.com/uc?id=1-0lkvwosd0l9j8_mcQKp4so1Irtg8YJl', MODEL_PATH, quiet=False)
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Prediction Function
def predict(image):
    img = np.array(image.resize((224, 224))) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    class_labels = ['Apple', 'Hourglass', 'Inverted Triangle', 'Pear', 'Rectangle']
    return class_labels[np.argmax(prediction)]

# Streamlit UI
st.title("👗 Body Shape Classifier")
st.write("Upload an image to predict the body shape.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button('Predict'):
        result = predict(image)
        st.success(f"Predicted Body Shape: **{result}**")
