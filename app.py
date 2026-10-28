import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="AesthetiScan AI - Medical Suite",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 AesthetiScan AI: Clinical Diagnostic & Remodeling Suite")
st.caption("Academic Medical School Prototype | Surface Hemodynamics & Dermal Texture Analysis")

def compute_clinical_maps(img_bgr):
    # Convert to YCrCb & LAB for quantitative color channel separation
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Erythema Map (a-channel highlights hemoglobin/vascular redness)
    erythema_map = cv2.applyColorMap(a, cv2.COLORMAP_JET)
    
    # Texture Map (Laplacian variance highlights micro-pores and fine line density)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_abs = cv2.convertScaleAbs(laplacian)
    texture_map = cv2.applyColorMap(laplacian_abs * 3, cv2.COLORMAP_HOT)
    
    # Metrics calculation
    norm_redness = min(100, max(0, int((np.mean(a) - 128) * 4)))
    norm_texture = min(100, max(0, int(np.var(laplacian) / 8)))
    norm_density = max(20, 100 - norm_texture)

    return {
        "erythema_map": cv2.cvtColor(erythema_map, cv2.COLOR_BGR2RGB),
        "texture_map": cv2.cvtColor(texture_map, cv2.COLOR_BGR2RGB),
        "metrics": {
            "Erythema Index": norm_redness,
            "Texture Variance": norm_texture,
            "Dermal Density": norm_density
        }
    }

def simulate_targeted_treatment(img_bgr, timeline_days, protocol):
    if timeline_days == 0:
        return img_bgr

    progress = timeline_days / 180.0
    
    # Edge-preserving guided filtering to target micro-texture while preserving facial edges
    d = int(7 + (6 * progress))
    smoothed = cv2.edgePreservingFilter(img_bgr, flags=1, sigma_s=15*progress+1, sigma_r=0.2)
    
    # Protocol-specific adjustments
    if protocol == "RF Microneedling":
        # Smooth micro-texture + slight sharpening for collagen tightness
        smoothed = cv2.addWeighted(smoothed, 0.85, img_bgr, 0.15, 0)
    elif protocol == "Biostimulator (Sculptra)":
        # Soft contrast enhancement for volumetric density feel
        lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.2 + (0.6 * progress), tileGridSize=(8,8))
        l = clahe.apply(l)
        smoothed = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        
    # Controlled natural blend
    result = cv2.addWeighted(img_bgr, 1.0 - (0.5 * progress), smoothed, 0.5 * progress, 0)
    return result

# Sidebar Controls
st.sidebar.header("📋 Patient Controls")
protocol = st.sidebar.selectbox(
    "Aesthetic Protocol",
    ["RF Microneedling", "Biostimulator (Sculptra)", "HA Dermal Fillers", "Neuromodulator"]
)

view_mode = st.sidebar.radio(
    "Visualization Mode",
    ["Simulated Progression", "Clinical Diagnostics (Heatmaps)"]
)

uploaded_file = st.file_uploader("Upload Facial Photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    analysis = compute_clinical_maps(img_bgr)

    # 1. Quantitative Diagnostics Header
    st.subheader("1. Quantitative Diagnostic Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Erythema Index (Redness)", f"{analysis['metrics']['Erythema Index']} / 100")
    col2.metric("Texture Variance (Pores)", f"{analysis['metrics']['Texture Variance']} / 100")
    col3.metric("Estimated Dermal Density", f"{analysis['metrics']['Dermal Density']} / 100")

    st.markdown("---")

    if view_mode == "Clinical Diagnostics (Heatmaps)":
        st.subheader("2. Surface & Vascular Heatmap Analysis")
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            st.image(analysis["erythema_map"], caption="Erythema / Vascular Concentration Map", use_container_width=True)
        with col_h2:
            st.image(analysis["texture_map"], caption="Surface Texture & Pore Variance Map", use_container_width=True)
            
    else:
        st.subheader("2. Longitudinal 180-Day Simulation Engine")
        days = st.slider("Timeline Post-Treatment:", 0, 180, 180, step=30, format="Day %d")
        
        simulated_bgr = simulate_targeted_treatment(img_bgr, days, protocol)
        simulated_rgb = cv2.cvtColor(simulated_bgr, cv2.COLOR_BGR2RGB)

        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.image(img_rgb, caption="Baseline (Day 0)", use_container_width=True)
        with col_v2:
            st.image(simulated_rgb, caption=f"Simulated Outcome ({protocol} - Day {days})", use_container_width=True)

else:
    st.info("👆 Upload a patient photograph to initiate clinical analysis.")
