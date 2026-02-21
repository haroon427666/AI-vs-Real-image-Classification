import streamlit as st
from PIL import Image
from model import load_model, predict_image
import torch

MODEL_PATH = "C:\Users\-\Desktop\.ipynb_checkpoints\Machine_learning\APPS\ai_vs_real_app\checkpoints\best_model.pth"

st.set_page_config(page_title="AI vs Real Detector", layout="centered")

st.title("🧠 AI vs Real Image Classifier")
st.write("Upload an image and check if it is AI-generated or Real.")

# Load model once
@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)

model = get_model()

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Analyzing..."):
            pred, conf = predict_image(model, image)

            if pred == 1:
                label = "🤖 AI Generated"
            else:
                label = "📷 Real Image"

            st.success(f"Prediction: {label}")
            st.info(f"Confidence: {conf*100:.2f}%")