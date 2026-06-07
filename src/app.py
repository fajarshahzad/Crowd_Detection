import sys
import os
import base64
from pathlib import Path
import streamlit as st
import cv2
import numpy as np
import time

# Add project root to python path to resolve relative imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit.components.v1 as components
from src.inference import CrowdInferencePipeline
from src.annotation.export import save_annotation_data

# Page configuration
st.set_page_config(
    page_title="CrowdSight AI - Dashboard",
    page_icon="CS",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Declare custom component
ANNOTATOR_DIR = str(PROJECT_ROOT / "src" / "annotation" / "annotator_component")
my_annotator = components.declare_component("my_annotator", path=ANNOTATOR_DIR)

# Load cached pipeline
@st.cache_resource
def get_pipeline(detect_path=None, segment_path=None):
    return CrowdInferencePipeline(detect_path, segment_path)

DETECT_MODEL_PATH = str(PROJECT_ROOT / "runs" / "detect" / "train" / "weights" / "best.pt")
SEGMENT_MODEL_PATH = str(PROJECT_ROOT / "runs" / "segment" / "train" / "weights" / "best.pt")
pipeline = get_pipeline(
    detect_path=DETECT_MODEL_PATH if Path(DETECT_MODEL_PATH).exists() else None,
    segment_path=SEGMENT_MODEL_PATH if Path(SEGMENT_MODEL_PATH).exists() else None
)

# Sidebar configurations
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="brand-mark">CS</div>
            <div>
                <h2>CrowdSight AI</h2>
                <p>Computer vision workspace</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    menu = st.radio(
        "Workspace",
        [
            "Dashboard",
            "Image Annotator",
            "Pedestrian Detection",
            "Crowd Classification",
            "Instance Segmentation"
        ]
    )
    st.markdown("---")
    st.markdown("### Preferences")
    theme_choice = st.selectbox("Theme", ["Dark", "Light"], index=0)
    st.markdown("### System")
    device_opt = st.selectbox("Device", ["CPU", "GPU (CUDA)"])

theme = {
    "Dark": {
        "bg": "#0b1020",
        "bg_soft": "#111827",
        "panel": "#151c2e",
        "panel_alt": "#0f172a",
        "text": "#f8fafc",
        "muted": "#94a3b8",
        "border": "rgba(148, 163, 184, 0.18)",
        "accent": "#38bdf8",
        "accent_2": "#22c55e",
        "shadow": "rgba(2, 6, 23, 0.42)",
        "input": "#0f172a",
    },
    "Light": {
        "bg": "#f6f8fb",
        "bg_soft": "#edf2f7",
        "panel": "#ffffff",
        "panel_alt": "#f8fafc",
        "text": "#111827",
        "muted": "#64748b",
        "border": "rgba(15, 23, 42, 0.10)",
        "accent": "#0284c7",
        "accent_2": "#16a34a",
        "shadow": "rgba(15, 23, 42, 0.10)",
        "input": "#ffffff",
    },
}[theme_choice]

# Inject modern theme-aware CSS stylesheet.
theme_css = """
<style>
    :root {{
        --cs-bg: %%BG%%;
        --cs-bg-soft: %%BG_SOFT%%;
        --cs-panel: %%PANEL%%;
        --cs-panel-alt: %%PANEL_ALT%%;
        --cs-text: %%TEXT%%;
        --cs-muted: %%MUTED%%;
        --cs-border: %%BORDER%%;
        --cs-accent: %%ACCENT%%;
        --cs-accent-2: %%ACCENT_2%%;
        --cs-shadow: %%SHADOW%%;
        --cs-input: %%INPUT%%;
    }}

    .stApp {
        background:
            radial-gradient(circle at top left, color-mix(in srgb, var(--cs-accent) 18%, transparent), transparent 32rem),
            linear-gradient(180deg, var(--cs-bg) 0%, var(--cs-bg-soft) 100%) !important;
        color: var(--cs-text) !important;
        font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1240px;
    }

    [data-testid="stSidebar"] {
        background: var(--cs-panel) !important;
        border-right: 1px solid var(--cs-border);
        box-shadow: 10px 0 30px var(--cs-shadow);
    }

    [data-testid="stSidebar"] * {
        color: var(--cs-text) !important;
    }

    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 0.9rem;
        padding: 0.35rem 0 1.1rem;
    }

    .brand-mark {
        width: 42px;
        height: 42px;
        border-radius: 8px;
        display: grid;
        place-items: center;
        background: linear-gradient(135deg, var(--cs-accent), var(--cs-accent-2));
        color: #fff !important;
        font-weight: 800;
        letter-spacing: 0;
        box-shadow: 0 10px 24px color-mix(in srgb, var(--cs-accent) 34%, transparent);
    }

    .sidebar-brand h2 {
        margin: 0;
        font-size: 1.08rem;
        line-height: 1.1;
    }

    .sidebar-brand p {
        margin: 0.22rem 0 0;
        color: var(--cs-muted) !important;
        font-size: 0.82rem;
    }

    .hero-card {
        background:
            linear-gradient(135deg, color-mix(in srgb, var(--cs-accent) 20%, var(--cs-panel)), var(--cs-panel) 58%),
            var(--cs-panel);
        border-radius: 8px;
        padding: 2rem;
        margin-bottom: 1.6rem;
        box-shadow: 0 22px 70px var(--cs-shadow);
        border: 1px solid var(--cs-border);
        position: relative;
        overflow: hidden;
    }

    .hero-card:after {
        content: "";
        position: absolute;
        inset: auto -8rem -9rem auto;
        width: 19rem;
        height: 19rem;
        border-radius: 50%;
        background: color-mix(in srgb, var(--cs-accent-2) 16%, transparent);
    }

    .hero-kicker {
        margin: 0 0 0.65rem;
        color: var(--cs-accent) !important;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .hero-card h1 {
        color: var(--cs-text) !important;
        font-size: clamp(2rem, 4vw, 3.7rem);
        line-height: 1;
        margin: 0;
        letter-spacing: 0;
        max-width: 800px;
    }

    .hero-card p {
        color: var(--cs-muted) !important;
        margin: 1rem 0 0;
        max-width: 720px;
        font-size: 1.02rem;
        line-height: 1.65;
    }
    
    .stat-card {
        background: color-mix(in srgb, var(--cs-panel) 94%, transparent);
        border-radius: 8px;
        padding: 1.35rem;
        border: 1px solid var(--cs-border);
        box-shadow: 0 16px 36px var(--cs-shadow);
        transition: transform 0.18s ease, border-color 0.18s ease;
        margin-bottom: 1.5rem;
    }
    .stat-card:hover {
        transform: translateY(-2px);
        border-color: color-mix(in srgb, var(--cs-accent) 55%, var(--cs-border));
    }
    .stat-value {
        font-size: 2rem;
        font-weight: 800;
        color: var(--cs-text);
        margin-bottom: 0.2rem;
        letter-spacing: 0;
    }
    .stat-label {
        font-size: 0.78rem;
        color: var(--cs-muted);
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .workflow-card {
        background: var(--cs-panel);
        border: 1px solid var(--cs-border);
        border-radius: 8px;
        padding: 1.25rem;
        box-shadow: 0 16px 36px var(--cs-shadow);
        min-height: 100%;
    }
    h1, h2, h3, h4, h5, h6, p, li, span, label {
        color: var(--cs-text) !important;
        letter-spacing: 0;
    }

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {
        color: var(--cs-muted) !important;
    }

    div[data-testid="stFileUploader"],
    div[data-testid="stCodeBlock"],
    div[data-testid="stAlert"],
    .stDataFrame,
    [data-testid="stImage"] {
        border-radius: 8px !important;
    }

    .stTextInput input,
    .stSelectbox div[data-baseweb="select"] > div,
    .stNumberInput input {
        background: var(--cs-input) !important;
        color: var(--cs-text) !important;
        border-color: var(--cs-border) !important;
        border-radius: 8px !important;
    }

    .stRadio [role="radiogroup"] {
        gap: 0.35rem;
    }

    .stRadio [role="radio"] {
        background: transparent;
        border-radius: 8px;
        padding: 0.58rem 0.7rem;
    }

    .stRadio [aria-checked="true"] {
        background: color-mix(in srgb, var(--cs-accent) 15%, transparent);
        border: 1px solid color-mix(in srgb, var(--cs-accent) 38%, transparent);
    }
    .stButton>button {
        background: var(--cs-panel) !important;
        color: var(--cs-text) !important;
        border: 1px solid var(--cs-border) !important;
        border-radius: 8px !important;
        transition: all 0.18s ease !important;
        box-shadow: 0 10px 24px var(--cs-shadow);
    }
    .stButton>button:hover {
        border-color: var(--cs-accent) !important;
        transform: translateY(-1px);
    }

    .stButton>button[kind="primary"] {
        background: linear-gradient(135deg, var(--cs-accent), var(--cs-accent-2)) !important;
        border-color: transparent !important;
        color: #fff !important;
    }

    hr {
        border-color: var(--cs-border) !important;
    }

    @media (max-width: 900px) {
        .hero-card {
            padding: 1.4rem;
        }
    }
</style>
"""
for token, value in {
    "%%BG%%": theme["bg"],
    "%%BG_SOFT%%": theme["bg_soft"],
    "%%PANEL%%": theme["panel"],
    "%%PANEL_ALT%%": theme["panel_alt"],
    "%%TEXT%%": theme["text"],
    "%%MUTED%%": theme["muted"],
    "%%BORDER%%": theme["border"],
    "%%ACCENT%%": theme["accent"],
    "%%ACCENT_2%%": theme["accent_2"],
    "%%SHADOW%%": theme["shadow"],
    "%%INPUT%%": theme["input"],
}.items():
    theme_css = theme_css.replace(token, value)
theme_css = theme_css.replace("{{", "{").replace("}}", "}")
st.markdown(theme_css, unsafe_allow_html=True)

# Custom Common CSS for Alert banners
st.markdown("""
<style>
    .alert-banner {
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        font-weight: 500;
        border-left: 5px solid;
    }
</style>
""", unsafe_allow_html=True)

# Global dataset paths
raw_img_path = PROJECT_ROOT / "Dataset" / "images"
processed_det = PROJECT_ROOT / "Dataset" / "processed" / "detection" / "train" / "images"
processed_seg = PROJECT_ROOT / "Dataset" / "processed" / "segmentation" / "train" / "images"

# Header Card
st.markdown(f"""
<div class="hero-card">
    <p class="hero-kicker">Crowd analytics platform</p>
    <h1>CrowdSight AI</h1>
    <p>Real-time crowd counting, density classification, bounding-box annotation, and instance segmentation in one focused workspace.</p>
</div>
""", unsafe_allow_html=True)

# ----------------- 1. DASHBOARD OVERVIEW -----------------
if menu == "Dashboard":
    st.subheader("System Overview")
    
    col1, col2, col3 = st.columns(3)
    num_raw = len(list(raw_img_path.iterdir())) if raw_img_path.exists() else 0
    num_det_split = len(list(processed_det.iterdir())) if processed_det.exists() else 0
    num_seg_split = len(list(processed_seg.iterdir())) if processed_seg.exists() else 0
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{num_raw}</div>
            <div class="stat-label">Raw Images in Dataset/images/</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{num_det_split} / {num_seg_split}</div>
            <div class="stat-label">Processed Split Images (Det/Seg)</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        status_label = "Ready" if num_det_split > 0 or num_seg_split > 0 else "Needs Splitting"
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{status_label}</div>
            <div class="stat-label">System Readiness</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("### Workflow")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Recommended Flow")
        st.markdown("""
        1. **Annotate**: Upload images and draw bounding boxes or polygon masks.
        2. **Train**: Customize hyperparameters and run the training scripts in your local console.
        3. **Evaluate**: Test detection, classification, and segmentation results interactively.
        """)
        
    with col_b:
        st.markdown("#### Dataset Snapshot")
        chart_data = {
            "Raw Images": num_raw,
            "Detection Split": num_det_split,
            "Segmentation Split": num_seg_split
        }
        st.bar_chart(chart_data)

# ----------------- 2. IMAGE ANNOTATOR -----------------
elif menu == "Image Annotator":
    st.subheader("Image Annotator")
    st.info("Upload an image, draw boxes or polygon masks, choose an export format, and save the finished annotations.")
    
    uploaded_file = st.file_uploader("Upload an image to annotate...", type=["jpg", "jpeg", "png", "bmp"])
    
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        
        # Base64 encode for component
        encoded_img = base64.b64encode(file_bytes).decode()
        data_url = f"data:image/jpeg;base64,{encoded_img}"
        
        # Display the custom annotator component
        drawn_anns = my_annotator(
            imageUrl=data_url,
            existing_annotations=[],
            key="annotator_board"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Export Settings")
            export_format = st.selectbox("Select Target Annotation Format", ["YOLO", "COCO", "COCO JSON"])
            
            save_to_raw = False
            if export_format == "YOLO":
                save_to_raw = st.checkbox("Also save to raw dataset (Dataset/images & Dataset/labels) for YOLO training", value=True)
            
            img_filename = st.text_input("Name of image file:", value=uploaded_file.name)
            
        with col2:
            st.markdown("### Save")
            st.write(f"Active shapes drawn: **{len(drawn_anns) if drawn_anns else 0}**")
            
            if st.button("Save Annotation", type="primary", key="save_ann_btn"):
                if drawn_anns:
                    success, msg = save_annotation_data(
                        image_name=img_filename,
                        image_bytes=file_bytes,
                        annotations=drawn_anns,
                        selected_format=export_format,
                        dataset_root=str(PROJECT_ROOT / "Dataset"),
                        save_to_raw=save_to_raw
                    )
                    if success:
                        st.success(msg)
                        
                        # Show confirmation preview of the saved annotated image
                        saved_ann_path = PROJECT_ROOT / "Dataset" / "annotations" / export_format.lower().replace(" ", "_") / "annotated" / f"{Path(img_filename).stem}_annotated.jpg"
                        if saved_ann_path.exists():
                            st.image(str(saved_ann_path), caption="Exported Annotated Image Preview", use_container_width=True)
                    else:
                        st.error(msg)
                else:
                    st.warning("Please draw at least one shape on the canvas and click 'Confirm' in the drawing board before saving.")

# ----------------- 3. PEDESTRIAN DETECTION -----------------
elif menu == "Pedestrian Detection":
    st.subheader("Pedestrian Detection")
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        conf_det = st.slider("Confidence Threshold (Lower detects hidden/sitting people)", min_value=0.01, max_value=1.00, value=0.15, step=0.01, key="conf_det_slider")
    with col_c2:
        iou_det = st.slider("IoU (NMS) Threshold (Higher preserves overlapping people)", min_value=0.05, max_value=0.95, value=0.60, step=0.05, key="iou_det_slider")

    uploaded_det_img = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png", "bmp"], key="det_img_upl")
    
    if uploaded_det_img is not None:
        det_bytes = np.asarray(bytearray(uploaded_det_img.read()), dtype=np.uint8)
        img = cv2.imdecode(det_bytes, 1)
        
        col_input, col_output = st.columns(2)
        with col_input:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Input Image", use_container_width=True)
            
        with col_output:
            with st.spinner("Executing pedestrian detection pipeline..."):
                start_time = time.time()
                res = pipeline.process_image(img, mode="detect", conf=conf_det, iou=iou_det)
                latency = (time.time() - start_time) * 1000
                
            st.image(res["annotated_image"], caption="Detected persons", use_container_width=True)
            st.markdown(f"""
            - **Detected people**: {res['count']}
            - **Detection Latency**: {latency:.1f} ms
            """)
            
    st.markdown("---")
    st.markdown("### Training Command")
    st.write("Customize the hyperparameters below to generate a local training command.")
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        epochs_det = st.number_input("Epochs", min_value=1, max_value=500, value=50, step=5, key="ep_det_console")
        model_size_det = st.selectbox("Base Model Size", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], key="sz_det_console")
    with col_t2:
        batch_det = st.selectbox("Batch Size", [4, 8, 16, 32, 64], index=2, key="bat_det_console")
        resume_det = st.checkbox("Auto-Resume training", value=True, key="res_det_console_chk")
        
    cmd_det = f"python src/detection/train_detection.py --epochs {epochs_det} --batch {batch_det} --model {model_size_det}"
    if resume_det:
        cmd_det += " --resume auto"
    
    st.write("Copy and run this command in your project terminal to train the detection model:")
    st.code(cmd_det, language="bash")

# ----------------- 4. CROWD CLASSIFICATION -----------------
elif menu == "Crowd Classification":
    st.subheader("Crowd Classification")
    
    st.markdown("### Density Thresholds")
    st.write("Configure the thresholds used to classify density and trigger recommendations.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        sparse_limit = st.slider("Sparse Limit (Max)", min_value=1, max_value=20, value=5)
    with col2:
        moderate_limit = st.slider("Moderate Limit (Max)", min_value=10, max_value=100, value=20)
    with col3:
        dense_limit = st.slider("Dense Limit (Max)", min_value=30, max_value=300, value=50)
        
    # Apply custom limits dynamically to classifier
    pipeline.classifier.sparse_limit = sparse_limit
    pipeline.classifier.moderate_limit = moderate_limit
    pipeline.classifier.dense_limit = dense_limit
    
    col_cc1, col_cc2 = st.columns(2)
    with col_cc1:
        conf_class = st.slider("Confidence Threshold (Lower detects hidden/sitting people)", min_value=0.01, max_value=1.00, value=0.15, step=0.01, key="conf_class_slider")
    with col_cc2:
        iou_class = st.slider("IoU (NMS) Threshold (Higher preserves overlapping people)", min_value=0.05, max_value=0.95, value=0.60, step=0.05, key="iou_class_slider")

    st.markdown("### Live Inference")
    uploaded_class_img = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png", "bmp"], key="class_img_upl")
    
    if uploaded_class_img is not None:
        class_bytes = np.asarray(bytearray(uploaded_class_img.read()), dtype=np.uint8)
        img = cv2.imdecode(class_bytes, 1)
        
        col_input, col_output = st.columns(2)
        with col_input:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Input Image", use_container_width=True)
            
        with col_output:
            with st.spinner("Processing deep learning crowd classification..."):
                res = pipeline.process_image(img, mode="detect", conf=conf_class, iou=iou_class)
                
            st.image(res["annotated_image"], caption="Classification Detection Map", use_container_width=True)
            
            cls_res = res["classification"]
            st.markdown(f"""
            <div class="alert-banner" style="background-color: {cls_res['hex_color']}22; color: {cls_res['hex_color']}; border-left-color: {cls_res['hex_color']};">
                <h4 style="margin: 0; color: {cls_res['hex_color']} !important;">Crowd Level Status: {cls_res['level']}</h4>
                <p style="margin: 0.5rem 0 0 0;"><b>Action Recommendation:</b> {cls_res['action']}</p>
                <p style="margin: 0.2rem 0 0 0; font-size: 0.9rem;">{cls_res['description']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            - **Detected people**: {res['count']}
            """)
            
            # Gauge chart visualization
            st.markdown("#### Threshold Meter")
            gauge_data = {
                "Sparse Safe Limit": sparse_limit,
                "Moderate Watch Limit": moderate_limit,
                "Dense Alert Limit": dense_limit,
                "Current Count": res["count"]
            }
            st.bar_chart(gauge_data)

# ----------------- 5. INSTANCE SEGMENTATION -----------------
elif menu == "Instance Segmentation":
    st.subheader("Instance Segmentation")
    
    col_cs1, col_cs2 = st.columns(2)
    with col_cs1:
        conf_seg = st.slider("Confidence Threshold (Lower detects hidden/sitting people)", min_value=0.01, max_value=1.00, value=0.15, step=0.01, key="conf_seg_slider")
    with col_cs2:
        iou_seg = st.slider("IoU (NMS) Threshold (Higher preserves overlapping people)", min_value=0.05, max_value=0.95, value=0.60, step=0.05, key="iou_seg_slider")

    uploaded_seg_img = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png", "bmp"], key="seg_img_upl")
    
    if uploaded_seg_img is not None:
        seg_bytes = np.asarray(bytearray(uploaded_seg_img.read()), dtype=np.uint8)
        img = cv2.imdecode(seg_bytes, 1)
        
        col_input, col_output = st.columns(2)
        with col_input:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Input Image", use_container_width=True)
            
        with col_output:
            with st.spinner("Executing instance segmentation mask generator..."):
                start_time = time.time()
                res = pipeline.process_image(img, mode="segment", conf=conf_seg, iou=iou_seg)
                latency = (time.time() - start_time) * 1000
                
            st.image(res["annotated_image"], caption="Segmented person masks", use_container_width=True)
            st.markdown(f"""
            - **Segmented people**: {res['count']}
            - **Segmentation Latency**: {latency:.1f} ms
            """)
            
    st.markdown("---")
    st.markdown("### Training Command")
    st.write("Customize the hyperparameters below to generate a local training command.")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        epochs_seg = st.number_input("Epochs", min_value=1, max_value=500, value=50, step=5, key="ep_seg_console")
        model_size_seg = st.selectbox("Base Model Size", ["yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt"], key="sz_seg_console")
    with col_s2:
        batch_seg = st.selectbox("Batch Size", [4, 8, 16, 32, 64], index=2, key="bat_seg_console")
        resume_seg = st.checkbox("Auto-Resume training", value=True, key="res_seg_console_chk")
        
    cmd_seg = f"python src/segmentation/train_segmentation.py --epochs {epochs_seg} --batch {batch_seg} --model {model_size_seg}"
    if resume_seg:
        cmd_seg += " --resume auto"
    
    st.write("Copy and run this command in your project terminal to train the segmentation model:")
    st.code(cmd_seg, language="bash")



