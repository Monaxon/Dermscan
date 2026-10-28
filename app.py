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
st.caption("Academic Medical School Prototype | Precision Dermal Neocollagenesis & Texture Engine")

def isolate_facial_skin_mask(img_bgr):
    # Convert to YCrCb for robust skin chromaticity detection
    img_ycbr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    
    # Tighter thresholds to exclude bright white clothing (fleece/jackets) and background
    min_YCrCb = np.array([40, 135, 85], np.uint8)
    max_YCrCb = np.array([240, 168, 122], np.uint8)
    skin_mask = cv2.inRange(img_ycbr, min_YCrCb, max_YCrCb)
    
    # Clean up noise using morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    
    # Keep only the largest central contour (assumed to be the face, ignoring background noise)
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        refined_mask = np.zeros_like(skin_mask)
        cv2.drawContours(refined_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        return refined_mask
    
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
    norm_texture = min(100, max(0, int(texture_score / 8)))
    norm_elasticity = max(15, 100 - norm_texture)
    
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

    # Smooth transition mask
    mask_float = cv2.GaussianBlur(skin_mask.astype(np.float32) / 255.0, (25, 25), 0)
    mask_3ch = cv2.merge([mask_float, mask_float, mask_float])

    # 1. Micro-texture refinement (simulating neocollagenesis / smoothing fine lines)
    sigma_s = int(10 * progress_factor) | 1
    sigma_r = 0.15 + (0.1 * progress_factor)
    smoothed = cv2.bilateralFilter(img_bgr, d=9, sigmaColor=75*sigma_r, sigmaSpace=75*sigma_s)

    # 2. Structural lifting effect for volumizing/tightening protocols
    if protocol in ["Biostimulator (Sculptra)", "RF Microneedling"]:
        shift_y = int(3 * progress_factor)
        M = np.float32([[1, 0, 0], [0, 1, -shift_y]])
        lifted = cv2.warpAffine(smoothed, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        smoothed = cv2.addWeighted(smoothed, 0.5, lifted, 0.5, 0)

    # Blend strictly over the isolated face mask, keeping natural skin tones intact
    remodeled_skin = (smoothed.astype(np.float32) * mask_3ch + img_bgr.astype(np.float32) * (1.0 - mask_3ch))
    remodeled_skin = np.clip(remodeled_skin, 0, 255).astype(np.uint8)

    # 3. Healthy dermal glow & subtle warmth enhancement in LAB color space
    lab = cv2.cvtColor(remodeled_skin, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Gentle contrast enhancement on the lightness channel
    clahe = cv2.createCLAHE(clipLimit=1.0 + (0.4 * progress_factor), tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Enhance warmth (b channel) slightly for a healthy post-treatment radiance
    b_enhanced = cv2.add(b, int(2 * progress_factor))
    
    enhanced_lab = cv2.merge((cl, a, b_enhanced))
    final_output = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # Final blend with original background to ensure absolute zero background distortion
    final_composite = (final_output.astype(np.float32) * mask_3ch + img_bgr.astype(np.float32) * (1.0 - mask_3ch))
    return np.clip(final_composite, 0, 255).astype(np.uint8)

st.sidebar.header("📋 Patient Data & Protocol")
selected_protocol = st.sidebar.selectbox(
    "Aesthetic Protocol",
    ["RF Microneedling", "Biostimulator (Sculptra)", "HA Dermal Fillers", "Neuromodulator"]
)

uploaded_file = st.file_uploader("Upload Front-Facing Facial Photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    skin_mask = isolate_facial_skin_mask(img_bgr)
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
