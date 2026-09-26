import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import joblib
import pandas as pd

st.set_page_config(page_title="Hydroponic AI System", layout="wide")

CLASSES = ['Healthy', 'K Deficient', 'N Deficient', 'Wilt fungal']
DEVICE = torch.device("cpu")

@st.cache_resource
def load_vision_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(CLASSES))
    model.load_state_dict(torch.load("models/baseline_mobilenetv2.pth", map_location=DEVICE))
    model.eval()
    return model

@st.cache_resource
def load_growth_model():
    return joblib.load("models/growth_predictor.pkl")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

st.title("🥬 Hydroponic AI: Vision + IoT Growth Monitoring System")
st.caption("Prototype system for AI-driven nutrient stress detection and growth prediction in indoor hydroponic farming — built for Gulf-region food security applications.")

tab1, tab2 = st.tabs(["🔍 Leaf Health Classifier", "🌱 Growth-Day Predictor"])

with tab1:
    st.subheader("Upload a lettuce leaf image")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", width=300)

        model = load_vision_model()
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)[0]
            pred_idx = torch.argmax(probs).item()

        st.success(f"Prediction: **{CLASSES[pred_idx]}** ({probs[pred_idx]*100:.1f}% confidence)")
        st.bar_chart({CLASSES[i]: probs[i].item() for i in range(len(CLASSES))})

with tab2:
    st.subheader("Predict growth day from sensor readings")
    col1, col2 = st.columns(2)
    with col1:
        temp = st.slider("Temperature (°C)", 18.0, 34.0, 28.0)
        humidity = st.slider("Humidity (%)", 50, 80, 65)
    with col2:
        tds = st.slider("TDS Value (ppm)", 300, 900, 600)
        ph = st.slider("pH Level", 5.5, 7.0, 6.3)

    if st.button("Predict Growth Day"):
        model = load_growth_model()
        features = pd.DataFrame([[temp, humidity, tds, ph]],
                                 columns=["Temperature (°C)", "Humidity", "TDS Value (ppm)", "pH Level"])
        pred = model.predict(features)[0]
        st.success(f"Predicted growth day: **{pred:.1f}** (of ~48-day cycle)")

st.divider()
st.caption("Prototype built as part of independent PhD-application research. Vision and sensor models are currently trained independently; multi-modal fusion is proposed future work.")