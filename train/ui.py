# ==============================
# OIL SPILL DETECTION UI (FINAL WORKING)
# ==============================

import streamlit as st
import torch
import numpy as np
import os
import cv2
from PIL import Image
import torchvision.transforms as transforms
from model import UNet

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Oil Spill Detection", layout="wide")

st.title("🛢️ Oil Spill Detection System")
st.write("Detect oil spill regions with RED overlay")

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, out_channels=1)

    # make path robust no matter where streamlit is launched from
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model, device

model, device = load_model()

# ==============================
# TRANSFORM
# ==============================
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),  # keep same preprocessing as training
])

# ==============================
# PREDICT FUNCTION
# ==============================
def predict(image):
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        raw_pred = output.squeeze().cpu().numpy()

    return raw_pred

# ==============================
# SIDEBAR SETTINGS
# ==============================
with st.sidebar:
    st.header("Settings")
    alpha = st.slider("Overlay Strength", 0.1, 0.9, 0.6, 0.1)
    threshold = st.slider("Mask Threshold", 0.1, 0.9, 0.5, 0.05)
    use_dynamic_threshold = st.checkbox("Use Dynamic Threshold (top 20%)", value=True)

# ==============================
# FILE UPLOAD
# ==============================
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# ==============================
# MAIN LOGIC
# ==============================
if uploaded_file is not None:

    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # ==============================
    # MODEL PREDICTION
    # ==============================
    raw_pred = predict(image)

    # Resize to original size
    raw_pred_resized = cv2.resize(raw_pred, (image_np.shape[1], image_np.shape[0]))

    # ==============================
    # MASK CREATION
    # ==============================
    # always compute a normalized map for visualization
    pred_min = float(raw_pred_resized.min())
    pred_max = float(raw_pred_resized.max())
    pred_range = pred_max - pred_min
    norm_pred = (raw_pred_resized - pred_min) / (pred_range + 1e-8)

    if use_dynamic_threshold and pred_range > 1e-6:
        # Dynamic threshold (top 20% pixels)
        threshold_dynamic = np.percentile(norm_pred, 80)
        pred_mask = (norm_pred > threshold_dynamic).astype(np.uint8)
    else:
        # Direct threshold on sigmoid output (0..1)
        pred_mask = (raw_pred_resized > threshold).astype(np.uint8)

    # ==============================
    # RED OVERLAY
    # ==============================
    overlay = image_np.copy()
    overlay[pred_mask == 1] = [255, 0, 0]
    # blend with original for adjustable strength
    overlay = (alpha * overlay + (1 - alpha) * image_np).astype(np.uint8)

    # ==============================
    # DISPLAY RESULTS
    # ==============================
    st.subheader("Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(image_np, caption="Original Image")

    with col2:
        st.image((pred_mask * 255).astype(np.uint8), caption="Predicted Mask")

    with col3:
        st.image(overlay, caption="Oil Spill Highlighted (RED)")

    # ==============================
    # DEBUG INFO
    # ==============================
    st.markdown("---")
    st.subheader("Debug Info")

    st.write("Max Raw:", pred_max)
    st.write("Min Raw:", pred_min)
    st.write("Mean Raw:", float(np.mean(raw_pred_resized)))

    # Show heatmap (VERY IMPORTANT)
    st.subheader("Model Heatmap")
    st.image(norm_pred, clamp=True)

    # ==============================
    # METRICS
    # ==============================
    oil_percent = np.mean(pred_mask) * 100
    st.metric("Oil Coverage", f"{oil_percent:.2f}%")
