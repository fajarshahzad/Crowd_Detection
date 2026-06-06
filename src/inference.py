import sys
import os
from pathlib import Path

# Add project root to python path to resolve relative imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from ultralytics import YOLO
from src.classification.classify_crowd import CrowdClassifier

class CrowdInferencePipeline:
    def __init__(self, detect_model_path=None, segment_model_path=None):
        fallback_detect = "yolov8n.pt"
        fallback_segment = "yolov8n-seg.pt"
        
        if detect_model_path and Path(detect_model_path).exists():
            self.detect_model = YOLO(detect_model_path)
            print(f"Loaded custom detection model from: {detect_model_path}")
        else:
            self.detect_model = YOLO(fallback_detect)
            print(f"Loaded default pretrained detection model: {fallback_detect}")
            
        if segment_model_path and Path(segment_model_path).exists():
            self.segment_model = YOLO(segment_model_path)
            print(f"Loaded custom segmentation model from: {segment_model_path}")
        else:
            self.segment_model = YOLO(fallback_segment)
            print(f"Loaded default pretrained segmentation model: {fallback_segment}")
            
        self.classifier = CrowdClassifier()

    def process_image(self, img_path_or_arr, mode="detect", conf=0.25, iou=0.7):
        if isinstance(img_path_or_arr, (str, Path)):
            img = cv2.imread(str(img_path_or_arr))
            if img is None:
                raise ValueError(f"Could not read image from {img_path_or_arr}")
        else:
            img = img_path_or_arr.copy()
            
        if mode == "detect":
            results = self.detect_model(img, classes=[0], conf=conf, iou=iou, verbose=False)
            result = results[0]
            annotated_img = result.plot()
            count = len(result.boxes) if result.boxes is not None else 0
            
        elif mode == "segment":
            results = self.segment_model(img, classes=[0], conf=conf, iou=iou, verbose=False)
            result = results[0]
            annotated_img = result.plot()
            count = len(result.boxes) if result.boxes is not None else 0
            
        else:
            raise ValueError(f"Unsupported mode: {mode}")
            
        annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        classification = self.classifier.classify(count)
        
        return {
            "annotated_image": annotated_rgb,
            "count": count,
            "classification": classification
        }
