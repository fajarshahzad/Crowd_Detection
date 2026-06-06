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
    page_icon="👥",
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
    st.markdown("<h2 style='text-align: center; color: #4e4376;'>👥 CrowdSight AI</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 0.9rem; color: #6c757d;'>Computer Vision Final Project Platform</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    menu = st.radio(
        "Navigation Hub",
        [
            "📊 Dashboard Overview",
            "✏️ Image Annotator",
            "🎯 Pedestrian Detection",
            "📈 Crowd Classification",
            "🎭 Instance Segmentation"
        ]
    )
    st.markdown("---")
    st.markdown("### System Configuration")
    device_opt = st.selectbox("Device", ["CPU", "GPU (CUDA)"])

# Inject Custom Light-Theme CSS Stylesheet
st.markdown("""
<style>
    .stApp {
        background-color: #f8f9fa !important;
        color: #212529 !important;
    }
    .header-card {
        background: linear-gradient(135deg, #2b5876 0%, #4e4376 100%);
        border-radius: 12px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        border: 1px solid rgba(0,0,0,0.05);
        text-align: center;
    }
    .header-card h1 { color: #ffffff !important; margin: 0; }
    .header-card p { color: #e9ecef !important; margin-top: 10px; }
    
    .stat-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 1.5rem;
        border: 1px solid #dee2e6;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        transition: transform 0.2s, box-shadow 0.2s;
        margin-bottom: 1.5rem;
    }
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-color: #ced4da;
    }
    .stat-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #4e4376;
        margin-bottom: 0.2rem;
    }
    .stat-label {
        font-size: 0.9rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    h1, h2, h3, h4, h5, h6, p, li, span, label {
        color: #212529 !important;
    }
    .stButton>button {
        background-color: #ffffff !important;
        color: #495057 !important;
        border: 1px solid #ced4da !important;
        border-radius: 6px !important;
        transition: all 0.2s !important;
    }
    .stButton>button:hover {
        background-color: #e9ecef !important;
        border-color: #adb5bd !important;
        color: #212529 !important;
    }
</style>
""", unsafe_allow_html=True)

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
<div class="header-card">
    <h1>👥 CrowdSight AI Platform</h1>
    <p>Premium Real-Time Crowd Counting, Density Classification, Bounding Box Annotation, and Instance Segmentation Platform</p>
</div>
""", unsafe_allow_html=True)

# ----------------- 1. DASHBOARD OVERVIEW -----------------
if menu == "📊 Dashboard Overview":
    st.subheader("📊 System Overview & Performance Metrics")
    
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
        
    st.markdown("### System Quick Start Dashboard")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Platform Workflow")
        st.markdown("""
        1. **Annotate**: Upload and mask/box objects using the **✏️ Image Annotator** tab. Export annotations to YOLO, COCO, or COCO JSON formats.
        2. **Train**: Customize hyperparameters and run the training scripts in your local console.
        3. **Inference**: Test pipeline results dynamically across Bounding Box or Instance Segmentation.
        """)
        
    with col_b:
        st.markdown("#### Database Quantity Visualizer")
        chart_data = {
            "Raw Images": num_raw,
            "Detection Split": num_det_split,
            "Segmentation Split": num_seg_split
        }
        st.bar_chart(chart_data)

# ----------------- 2. IMAGE ANNOTATOR -----------------
elif menu == "✏️ Image Annotator":
    st.subheader("✏️ Interactive Multi-Format Image Annotator")
    st.info("Upload an image, draw Bounding Boxes or Polygon Masks, select the export format, and save annotations. The annotated image with overlay labels/masks will automatically be saved.")
    
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
            st.markdown("### Export Directory Configurations")
            export_format = st.selectbox("Select Target Annotation Format", ["YOLO", "COCO", "COCO JSON"])
            
            save_to_raw = False
            if export_format == "YOLO":
                save_to_raw = st.checkbox("Also save to raw dataset (Dataset/images & Dataset/labels) for YOLO training", value=True)
            
            img_filename = st.text_input("Name of image file:", value=uploaded_file.name)
            
        with col2:
            st.markdown("### Save Annotations")
            st.write(f"Active shapes drawn: **{len(drawn_anns) if drawn_anns else 0}**")
            
            if st.button("💾 Save Annotation to Disk", type="primary", key="save_ann_btn"):
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
elif menu == "🎯 Pedestrian Detection":
    st.subheader("🎯 Pedestrian Detection Module (YOLOv8)")
    
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
                
            st.image(res["annotated_image"], caption="Detected Persons Bounding Box", use_container_width=True)
            st.markdown(f"""
            - **Detected Persons**: {res['count']}
            - **Detection Latency**: {latency:.1f} ms
            """)
            
    st.markdown("---")
    st.markdown("### Console Training Command Generator")
    st.write("Customize your hyperparameters below to generate the console training command:")
    
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
elif menu == "📈 Crowd Classification":
    st.subheader("📈 Crowd Density Classification Module")
    
    st.markdown("### Dynamic Crowd Threshold Configuration")
    st.write("Configure the thresholds that classify density and trigger specific crowd-control warnings.")
    
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

    st.markdown("### Live Classification Inference")
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
                <p style="margin: 0.2rem 0 0 0; font-size: 0.9rem; color: #8b949e;">{cls_res['description']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            - **Detected Persons**: {res['count']}
            """)
            
            # Gauge chart visualization
            st.markdown("#### Threshold Gauge Meter")
            gauge_data = {
                "Sparse Safe Limit": sparse_limit,
                "Moderate Watch Limit": moderate_limit,
                "Dense Alert Limit": dense_limit,
                "Current Count": res["count"]
            }
            st.bar_chart(gauge_data)

# ----------------- 5. INSTANCE SEGMENTATION -----------------
elif menu == "🎭 Instance Segmentation":
    st.subheader("🎭 Instance Segmentation Module (YOLOv8-seg)")
    
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
                
            st.image(res["annotated_image"], caption="Segmented Person Masks", use_container_width=True)
            st.markdown(f"""
            - **Segmented Persons**: {res['count']}
            - **Segmentation Latency**: {latency:.1f} ms
            """)
            
    st.markdown("---")
    st.markdown("### Console Training Command Generator")
    st.write("Customize your hyperparameters below to generate the console training command:")
    
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


