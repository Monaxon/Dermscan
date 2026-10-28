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

def compute_skin_metrics(img_bgr):
    b, g, r = cv2.split(img_bgr)
    redness_index = float(np.mean(r) / (np.mean(g) + 1e-5))
    
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture_score = float(np.var(laplacian))

    norm_redness = min(100, max(0, int((redness_index - 0.9) * 200)))
    norm_texture = min(100, max(0, int(texture_score / 10)))
    norm_elasticity = max(15, 100 - norm_texture)
    
    return {
        "Erythema Index": norm_redness,
        "Surface Texture Variance": norm_texture,
        "Estimated Dermal Density": norm_elasticity
    }

def simulate_collagen_remodeling(img_bgr, timeline_days, protocol):
    if timeline_days == 0:
        return img_bgr

    progress_factor = timeline_days / 180.0
    h, w, c = img_bgr.shape

    # 1. Advanced bilateral filtering for realistic pore and micro-texture refinement
    d_val = int(5 + (5 * progress_factor))
    sigma_color = 50 + (30 * progress_factor)
    sigma_space = 50 + (30 * progress_factor)
    
    smoothed = cv2.bilateralFilter(img_bgr, d=d_val, sigmaColor=sigma_color, sigmaSpace=sigma_space)

    # 2. Subtle volumetric tightening/lifting effect for clinical simulation
    if protocol in ["Biostimulator (Sculptra)", "RF Microneedling"]:
        shift_y = int(2 * progress_factor)
        M = np.float32([[1, 0, 0], [0, 1, -shift_y]])
        lifted = cv2.warpAffine(smoothed, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        smoothed = cv2.addWeighted(smoothed, 0.6, lifted, 0.4, 0)

    # 3. Enhance radiance and natural skin warmth in LAB color space (prevents any black-and-white washout)
    lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Controlled contrast boost
    clahe = cv2.createCLAHE(clipLimit=1.0 + (0.5 * progress_factor), tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Enrich the color warmth channel slightly for a healthy post-treatment glow
    b_chan = cv2.add(b, int(3 * progress_factor))
    
    enhanced_lab = cv2.merge((cl, a, b_chan))
    final_output = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # Blend with original to keep it entirely natural and realistic
    blended = cv2.addWeighted(img_bgr, 1.0 - (0.4 * progress_factor), final_output, 0.4 * progress_factor, 0)
    return blended

st.sidebar.header("📋 Patient Data & Protocol")
selected_protocol = st.sidebar.selectbox(
    "Aesthetic Protocol",
    ["RF Microneedling", "Biostimulator (Sculptra)", "HA Dermal Fillers", "Neuromodulator"]
)

uploaded_file = st.file_uploader("Upload Front-Facing Facial Photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    metrics = compute_skin_metrics(img_bgr)

    st.subheader("1. Quantitative Diagnostic Baseline")
    col1, col2, col3 = st.columns(3)
    col1.metric("Erythema Index (Redness)", f"{metrics['Erythema Index']} / 100")
    col2.metric("Texture Variance (Pores)", f"{metrics['Surface Texture Variance']} / 100")
    col3.metric("Estimated Dermal Density", f"{metrics['Estimated Dermal Density']} / 100")

    st.markdown("---")
    st.subheader("2. Longitudinal 180-Day Simulation Engine")

    days = st.slider("Select Timeline Post-Treatment:", 0, 180, 180, step=30, format="Day %d")

    simulated_bgr = simulate_collagen_remodeling(img_bgr, days, selected_protocol)

    img_rgb_baseline = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    simulated_rgb = cv2.cvtColor(simulated_bgr, cv2.COLOR_BGR2RGB)

    view_col1, view_col2 = st.columns(2)
    with view_col1:
        st.image(img_rgb_baseline, caption="Baseline Image (Day 0)", use_container_width=True)
    with view_col2:
        st.image(simulated_rgb, caption=f"Simulated Result ({selected_protocol} - Day {days})", use_container_width=True)
else:
    st.info("👆 Please upload a photo to begin the analysis.")
