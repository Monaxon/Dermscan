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

    # 1. Pronounced texture smoothing simulating collagen density increase over 180 days
    sigma_val = int(8 * progress_factor)
    smoothed = cv2.detailEnhance(img_bgr, sigma_s=10 + sigma_val, sigma_r=0.15 + (0.15 * progress_factor))

    # 2. Stronger bilateral smoothing to clear micro-imperfections and pores
    smoothed = cv2.bilateralFilter(smoothed, d=9, sigmaColor=75, sigmaSpace=75)

    # 3. Clinical lifting / tightening warp effect for volumetric stimulation
    if protocol in ["Biostimulator (Sculptra)", "RF Microneedling"]:
        shift_y = int(4 * progress_factor)
        M = np.float32([[1, 0, 0], [0, 1, -shift_y]])
        lifted = cv2.warpAffine(smoothed, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        smoothed = cv2.addWeighted(smoothed, 0.5, lifted, 0.5, 0)

    # 4. Vibrant clinical glow and warmth enhancement in LAB color space
    lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Noticeable brightening and clarity boost
    clahe = cv2.createCLAHE(clipLimit=1.5 + (1.0 * progress_factor), tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Add healthy post-treatment radiance (warmth)
    b_chan = cv2.add(b, int(6 * progress_factor))
    
    enhanced_lab = cv2.merge((cl, a, b_chan))
    final_output = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # Strong blend ratio so the 180-day transformation is striking and clear
    blend_weight = 0.2 + (0.6 * progress_factor)
    blended = cv2.addWeighted(img_bgr, 1.0 - blend_weight, final_output, blend_weight, 0)
    
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
