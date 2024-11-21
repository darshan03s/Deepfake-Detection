import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from download_model import download_model

st.set_page_config(page_title="Deepfake Detection | Image", layout="centered",initial_sidebar_state="auto", menu_items=None)

with st.spinner("Downloading model..."):
    download_model()

@st.cache_resource
def load_model():
    """Load the pre-trained deepfake detection model"""
    try:
        model = tf.keras.models.load_model('./DD_Image.h5')
        return model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

# Load the model
with st.spinner("Loading model..."):
    model = load_model()

if model is None:
    st.stop()  # Stop further execution if model loading fails

def preprocess_image(image):
    """Preprocess the uploaded image for prediction"""
    img = Image.fromarray(image, 'RGB')
    img = img.resize((128, 128))
    img_array = np.array(img)
    input_img = np.expand_dims(img_array, axis=0)
    return input_img

# Streamlit UI
st.title("Image Deepfake Detection")
st.markdown("Upload an image to detect if it is real or fake.")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Display the uploaded image
    st.image(img_rgb, caption="Uploaded Image", use_container_width=True)
    
    # Preprocess and predict
    input_img = preprocess_image(img_rgb)
    predictions = model.predict(input_img)
    
    # Get the prediction
    class_index = np.argmax(predictions, axis=1)[0]
        
    # Interpret the results
    result = 'FAKE' if class_index == 0 else 'REAL'
    
    # Display results
    st.write(f"### Prediction: {result}")