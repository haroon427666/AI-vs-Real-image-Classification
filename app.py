import os
import streamlit as st
import gdown
from PIL import Image
from model import load_model, predict_image

MODEL_PATH = "best_model.pth"
FILE_ID = "1pk7siHLeWZaiafyVviSR3nNAS4gJ5qGh"
GDRIVE_URL = f"https://drive.google.com/uc?id={FILE_ID}"

# Download model if needed,
if not os.path.exists(MODEL_PATH):
    with st.spinner("⬇️ Downloading model..."):
        gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

st.set_page_config(page_title="AI vs Real Detector")

st.title("🧠 AI vs Real Image Classifier")

@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)

model = get_model()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg", "webp"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("🔍 Predicting..."):
            pred, conf = predict_image(model, image)

            if pred == 1:
                st.success(f"🤖 AI Generated ({conf*100:.2f}%)")
            else:
                st.success(f"📷 Real Image ({conf*100:.2f}%)")
