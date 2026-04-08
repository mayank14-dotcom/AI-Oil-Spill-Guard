import streamlit as st
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os

from model import UNet

st.set_page_config(page_title="Oil Spill Detection System", layout="wide")


@st.cache_resource
def load_model():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UNet(in_channels=3, out_channels=1)
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model.pth")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model, device
    except FileNotFoundError:
        st.error("Model file 'best_model.pth' not found. Please train the model first using train.py.")
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None


model, device = load_model()


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


def create_overlay(original_image, mask, alpha=0.5):
    image_np = np.array(original_image)
    colored_mask = np.zeros_like(image_np)
    colored_mask[:, :, 0] = mask * 255
    overlay = image_np * (1 - alpha) + colored_mask * alpha
    return overlay.astype(np.uint8)


UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploaded_images")
os.makedirs(UPLOAD_DIR, exist_ok=True)


st.title("Oil Spill Detection using Deep Learning")
st.write(
    "Upload a satellite image and the system will detect oil spill regions "
    "using a trained U-Net deep learning model."
)


with st.sidebar:
    st.header("Settings")
    alpha = st.slider("Overlay Strength", 0.1, 0.9, 0.6, 0.1)
    threshold = st.slider("Mask Threshold", 0.1, 0.9, 0.5, 0.05)
    use_dynamic_threshold = st.checkbox("Use Dynamic Threshold (top 20%)", value=True)


uploaded_file = st.file_uploader("Upload Satellite Image", type=["jpg", "png", "jpeg"])

if uploaded_file and model is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # save uploaded image
    image.save(os.path.join(UPLOAD_DIR, uploaded_file.name))

    st.subheader("Original Satellite Image")
    st.image(image, use_container_width=True)

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(input_tensor)

    mask = prediction.squeeze().detach().cpu().numpy()

    if use_dynamic_threshold and (mask.max() - mask.min()) > 1e-6:
        norm = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        th = np.percentile(norm, 80)
        binary_mask = (norm > th).astype(np.uint8)
    else:
        binary_mask = (mask > threshold).astype(np.uint8)

    overlay_image = create_overlay(image, binary_mask, alpha=alpha)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Original Image")
        st.image(image)

    with col2:
        st.subheader("Predicted Mask")
        st.image(binary_mask * 255)

    with col3:
        st.subheader("Highlighted Oil Spill Region")
        st.image(overlay_image)

    spill_pixels = np.sum(binary_mask)
    total_pixels = binary_mask.size
    spill_percentage = (spill_pixels / total_pixels) * 100

    st.subheader("Oil Spill Statistics")
    st.write(f"Oil Spill Pixels: {spill_pixels}")
    st.write(f"Total Pixels: {total_pixels}")
    st.write(f"Estimated Spill Coverage: {spill_percentage:.2f}%")

    overlay_pil = Image.fromarray(overlay_image)
    st.download_button(
        label="Download Result Image",
        data=overlay_pil.tobytes(),
        file_name="oil_spill_result.png",
    )
