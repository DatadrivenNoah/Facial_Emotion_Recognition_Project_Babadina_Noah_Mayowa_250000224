import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from datetime import datetime
import tensorflow as tf

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="Emotion Detector",
    layout="centered",
    page_icon="😊"
)

# -----------------------------
# STYLING - Glassmorphism
# -----------------------------
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(135deg, #74ebd5, #ACB6E5);
        font-family: 'Arial', sans-serif;
    }
    .glass {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='glass'>", unsafe_allow_html=True)
st.title("😊 Emotion Detector")
st.write("Upload a photo or take a live picture to detect emotion!")

# -----------------------------
# FILE UPLOADER / CAMERA
# -----------------------------
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
camera_file = st.camera_input("Or take a photo with your webcam")
image_file = uploaded_file if uploaded_file is not None else camera_file

# -----------------------------
# EMOTION MODEL LOADING
# -----------------------------
# Replace with your own model path
model = tf.keras.models.load_model("emotion_model.h5")  

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
EMOTION_DETAILS = {
    "Angry": ("Feeling angry", "Try to relax and take deep breaths."),
    "Disgust": ("Feeling disgusted", "Take a moment to step away."),
    "Fear": ("Feeling scared", "Focus on safety and calm your mind."),
    "Happy": ("Feeling happy", "Keep enjoying the moment!"),
    "Sad": ("Feeling sad", "Reach out to friends or relax."),
    "Surprise": ("Feeling surprised", "Embrace the unexpected!"),
    "Neutral": ("Feeling neutral", "Maintain your balanced mood.")
}

# -----------------------------
# IMAGE PROCESSING FUNCTION
# -----------------------------
def preprocess_image(img):
    img = ImageOps.grayscale(img)  # convert to grayscale if model requires
    img = img.resize((48, 48))      # resize to model input size
    img_array = np.array(img) / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=-1)  # add channel dimension
    img_array = np.expand_dims(img_array, axis=0)   # add batch dimension
    return img_array

# -----------------------------
# PREDICTION
# -----------------------------
if image_file is not None:
    try:
        image = Image.open(image_file)
        st.image(image, caption="Input Image", use_column_width=True)

        img_array = preprocess_image(image)

        with st.spinner("🔍 Predicting emotion..."):
            predictions = model.predict(img_array, verbose=0)
            emotion_index = int(np.argmax(predictions))
            emotion = EMOTIONS[emotion_index]
            confidence = float(predictions[0][emotion_index] * 100)
        
        description, recommendation = EMOTION_DETAILS[emotion]

        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.success(f"**Emotion:** {emotion}")
        st.info(f"**Confidence:** {confidence:.2f}%")
        st.write(description)
        st.write("👉 " + recommendation)
        st.caption(f"Predicted at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error("❌ Error processing the image.")
        st.exception(e)

st.markdown("</div>", unsafe_allow_html=True)

