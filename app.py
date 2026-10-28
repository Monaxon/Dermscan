import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="AesthetiScan AI - Medical Prototype",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 AesthetiScan AI: Clinical Diagnostic & 180-Day Simulator")
st.caption("Academic Medical School Prototype | Skin Metrics & Dermal Neocollagenesis Engine")

def isolate_skin_mask(img_bgr):
    img_ycbr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    min_YCrCb = np.array([0, 133, 77], np.uint8)
    max_YCrCb = np.array([255, 173, 127], np.uint8)
    skin_mask = cv2.inRange(img_ycbr, min_YCrCb, max_YCrCb)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    return skin_mask

def compute_skin_metrics(img_bgr, skin_mask):
    b, g, r = cv2.split(img_bgr)
    skin_pixels_r = r[skin_mask > 0]
    skin_pixels_g = g[skin_mask > 0]
    
    if len(skin_pixels_r) > 0:
        redness_index = float(np.mean(skin_pixels_r) / (np.mean(skin_pixels_g) + 1e-5))
    else:
        redness_index = 1.0

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    skin_laplacian = laplacian[skin_mask > 0]
    
    if len(skin_laplacian) > 0:
        texture_score = float(np.var(skin_laplacian))
    else:
        texture_score = 100.0

    norm_redness = min(100, max(0, int((redness_index - 0.9) * 200)))
    norm_texture = min(100, max(0, int(texture_score / 10)))
    norm_elasticity = max(10, 100 - norm_texture)
    
    return {
        "Erythema Index": norm_redness,
        "Surface Texture Variance": norm_texture,
        "Estimated Dermal Density": norm_elasticity
    }

def simulate_collagen_remodeling(img_bgr, skin_mask, timeline_days, protocol):
    if timeline_days == 0:
        return img_bgr

    progress_factor = timeline_days / 180.0
    h, w, c = img_bgr.shape
    output = img_bgr.copy()

    mask_float = cv2.GaussianBlur(skin_mask.astype(np.float32) / 255.0, (15, 15), 0)
    mask_3ch = cv2.merge([mask_float, mask_float, mask_float])

    dsigma_s = int(15 * progress_factor) | 1
    dsigma_r = 0.1 + (0.2 * progress_factor)
    smoothed = cv2.edgePreservingFilter(img_bgr, flags=1, sigma_s=dsigma_s, sigma_r=dsigma_r)

    if protocol in ["Biostimulator (Sculptra)", "RF Microneedling"]:
        shift_y = int(4 * progress_factor)
        M = np.float32([[1, 0, 0], [0, 1, -shift_y]])
        lifted = cv2.warpAffine(smoothed, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        smoothed = cv2.addWeighted(smoothed, 0.4, lifted, 0.6, 0)

    remodeled_skin = (smoothed * mask_3ch + img_bgr * (1.0 - mask_3ch)).astype(np.uint8)

    lab = cv2.cvtColor(remodeled_skin, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.0 + (0.8 * progress_factor), tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    final_output = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    return final_output

st.sidebar.header("📋 Patient Data & Protocol")
selected_protocol = st.sidebar.selectbox(
    "Aesthetic Protocol",
    ["RF Microneedling", "Biostimulator (Sculptra)", "HA Dermal Fillers", "Neuromodulator"]
)

uploaded_file = st.file_uploader("Upload Front-Facing Facial Photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    skin_mask = isolate_skin_mask(img_bgr)
    metrics = compute_skin_metrics(img_bgr, skin_mask)

    st.subheader("1. Quantitative Diagnostic Baseline")
    col1, col2, col3 = st.columns(3)
    col1.metric("Erythema Index (Redness)", f"{metrics['Erythema Index']} / 100")
    col2.metric("Texture Variance (Pores)", f"{metrics['Surface Texture Variance']} / 100")
    col3.metric("Estimated Dermal Density", f"{metrics['Estimated Dermal Density']} / 100")

    st.markdown("---")
    st.subheader("2. Longitudinal 180-Day Simulation Engine")

    days = st.slider("Select Timeline Post-Treatment:", 0, 180, 180, step=30, format="Day %d")

    simulated_bgr = simulate_collagen_remodeling(img_bgr, skin_mask, days, selected_protocol)

    img_rgb_baseline = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    simulated_rgb = cv2.cvtColor(simulated_bgr, cv2.COLOR_BGR2RGB)

    view_col1, view_col2 = st.columns(2)
    with view_col1:
        st.image(img_rgb_baseline, caption="Baseline Image (Day 0)", use_container_width=True)
    with view_col2:
        st.image(simulated_rgb, caption=f"Simulated Result ({selected_protocol} - Day {days})", use_container_width=True)
else:
    st.info("👆 Please upload a photo to begin the analysis.")
  
