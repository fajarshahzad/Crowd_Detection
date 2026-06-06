# CrowdSight AI - Crowd Analytics & Multi-Format Annotator Platform

Welcome to **CrowdSight AI**, a premium, state-of-the-art computer vision platform designed for real-time crowd analytics, pedestrian bounding box detection, crowd density classification, instance mask segmentation, and interactive multi-format image annotations.

---

## 📁 Dataset Source Information

The dataset images used in this project (filenames starting with prefixes `273271-` and `273275-`) are sourced from **CrowdHuman**, a benchmark dataset for detecting humans in crowded and heavily occluded scenes.

- **Official CrowdHuman Project Page**: [CrowdHuman Website](https://www.crowdhuman.org/)
- **Crowd Counting Datasets**: https://www.kaggle.com/datasets/permanalwep/crowdhuman-crowd-detection

---

## ⚙️ Installation & Setup (Step-by-Step)

Follow these steps to set up and run the project locally on your system:
```
### Step 1: Install Required Dependencies
Install the necessary deep learning and web interface packages:
```bash
pip install streamlit ultralytics opencv-python numpy pillow
```

### Step 2: Verify Dataset Directory Structure
Ensure your datasets are placed in the correct directories:
```
Final_Project/
├── Dataset/
│   ├── images/       <- Place raw training images (.jpg) here
│   └── labels/       <- Place corresponding YOLO label text (.txt) here
```

### Step 3: Start the Platform
Launch the Streamlit web dashboard:
```bash
streamlit run src/app.py
```
This will automatically open the platform inside your default web browser (typically at `http://localhost:8501`).

---

## 🛠️ Platform Navigation & Features

### 1. 📊 Dashboard Overview
Provides a real-time status of your system readiness, the number of raw images available in your dataset, and charts showing active image split quantities.

### 2. ✏️ Image Annotator (Labels Masking & Exports)
- **Upload Image**: Drag and drop any `.jpg`, `.jpeg`, or `.png` image.
- **Draw Annotations**: 
  - Select **Bounding Box** to click-and-drag boxes.
  - Select **Polygon Mask** to click multiple vertices (double-click or close to finalize the mask).
- **Export Format Dropdown**: Select between **YOLO**, **COCO**, or **COCO JSON** formats.
- **Save to Disk**: Click **Save Annotation to Disk**. 
  - Annotations are saved in `Dataset/annotations/<selected_format>/`.
  - The annotated image with visual label/mask overlays is automatically saved under the respective `annotated/` folder for review.

### 3. 🎯 Pedestrian Detection & 🎭 Instance Segmentation
- **Interactive Threshold Sliders**: Adjust confidence and IoU (NMS) thresholds. Lowering the confidence threshold (e.g. `0.15`) detects sitting/hidden people, while raising the IoU (NMS) threshold (e.g. `0.60`) preserves overlapping individuals in dense crowds.
- **Command Generator**: Customize **Epochs**, **Batch Size**, and **Base weights** in the training generator to output the exact command block. Copy and run the generated command directly in your console.

### 4. 📈 Crowd Classification
Classifies crowd density levels into **Sparse (Safe)**, **Moderate (Watch)**, **Dense (Alert)**, or **Crowded (Critical)** based on dynamic threshold sliders, outputting safety recommendations.

---

## 📝 Console Training Commands

If you wish to train the models directly in your terminal, run the following commands:

- **Bounding Box Detection Model Training**:
  ```bash
  python src/detection/train_detection.py --epochs 50 --batch 16 --model yolov8n.pt
  ```
- **Instance Segmentation Model Training**:
  ```bash
  python src/segmentation/train_segmentation.py --epochs 50 --batch 16 --model yolov8n-seg.pt
  ```
