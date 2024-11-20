import streamlit as st
import numpy as np
import os
import cv2
import tensorflow as tf

st.set_page_config(page_title="Deepfake Detection", layout="centered",initial_sidebar_state="auto", menu_items=None)

upload_dir = "uploads"
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(
            'DD_gru_model_epoch(01)_acc(0.82)_valacc(0.83)_loss(0.69)_valloss(0.69)_07-11-24 20-23-07.keras')
        st.success("Model loaded successfully")
        return model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

# Load the model
with st.spinner("Loading model..."):
    model = load_model()

if model is None:
    st.stop()  # Stop further execution if model loading fails

IMG_SIZE = 224
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048


def build_feature_extractor():
    feature_extractor = tf.keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )

    preprocess_input = tf.keras.applications.inception_v3.preprocess_input
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return tf.keras.Model(inputs, outputs, name="feature_extractor_iv3")


feature_extractor = build_feature_extractor()


def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)

    return frame[start_y: start_y+min_dim, start_x: start_x+min_dim]


def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    frames = []

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break

    except Exception as e:
        st.error(e)

    finally:
        cap.release()

    return np.array(frames)


def prepare_single_video(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(
        shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(
                batch[None, j, :], verbose=0)
        frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked
    return frame_features, frame_mask


# Streamlit UI
st.title("Deepfake Detection with Video Upload")
st.markdown("Upload a video to predict if it is real or fake.")

video_file = st.file_uploader("Choose a video file", type=["mp4", "mov"])

if video_file is not None:
    # Save the uploaded video file
    video_path = os.path.join("uploads", video_file.name)
    with open(video_path, "wb") as f:
        f.write(video_file.getbuffer())
    
    st.video(video_path)  # Show the uploaded video
    
    frames = load_video(video_path)
    frame_features, frame_mask = prepare_single_video(frames)
    
    # Predict with the model
    prediction = model.predict([frame_features, frame_mask])[0]
    result = 'FAKE' if prediction >= 0.50 else 'REAL'
    confidence = float(prediction)
    
    st.write(f"Prediction: {result}")
    st.write(f"Confidence: {confidence:.2f}")
    
    # Optionally remove the uploaded video after prediction
    # os.remove(video_path)
