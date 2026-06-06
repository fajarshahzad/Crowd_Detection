import os
import json
import cv2
import numpy as np
from pathlib import Path

def get_class_id(label, class_mapping):
    if label not in class_mapping:
        # Assign a new class ID if not exists
        if class_mapping:
            class_mapping[label] = max(class_mapping.values()) + 1
        else:
            class_mapping[label] = 0
    return class_mapping[label]

def draw_annotations(img, annotations, class_mapping):
    """
    Renders bounding boxes and polygons on the image for preview/export.
    """
    img_h, img_w, _ = img.shape
    overlay = img.copy()
    
    # Predefined colors for drawing
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
    ]
    
    for ann in annotations:
        label = ann.get("label", "person")
        cid = get_class_id(label, class_mapping)
        color = colors[cid % len(colors)]
        
        if ann["type"] == "bbox":
            x_min = int(ann["x"] * img_w)
            y_min = int(ann["y"] * img_h)
            w = int(ann["w"] * img_w)
            h = int(ann["h"] * img_h)
            
            # Draw rectangle
            cv2.rectangle(img, (x_min, y_min), (x_min + w, y_min + h), color, 2)
            cv2.putText(img, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
        elif ann["type"] == "polygon":
            points = ann["points"]
            pts = np.array([[int(p[0] * img_w), int(p[1] * img_h)] for p in points], dtype=np.int32)
            if len(pts) > 0:
                # Fill polygon mask with transparency
                cv2.fillPoly(overlay, [pts], color)
                # Draw boundary lines
                cv2.polylines(img, [pts], True, color, 2)
                # Label at first point
                cv2.putText(img, label, (pts[0][0], pts[0][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
    # Blend overlay with 30% alpha for transparency
    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
    return img

def save_annotation_data(image_name, image_bytes, annotations, selected_format, dataset_root, save_to_raw=False):
    """
    Saves annotations and original image to format-specific directories.
    Also generates and saves an annotated version.
    """
    dataset_root = Path(dataset_root)
    class_mapping = {"person": 0}  # default mapping
    
    # 1. Decode original image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return False, "Could not decode uploaded image."
    img_h, img_w, _ = img.shape
    
    # Setup base output directories
    base_dir = dataset_root / "annotations"
    
    # Process files based on selected format
    if selected_format == "YOLO":
        # Paths
        yolo_dir = base_dir / "yolo"
        images_dir = yolo_dir / "images"
        labels_dir = yolo_dir / "labels"
        annotated_dir = yolo_dir / "annotated"
        
        # Create directories
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        annotated_dir.mkdir(parents=True, exist_ok=True)
        
        # Paths for specific files
        img_name_path = Path(image_name)
        txt_name = f"{img_name_path.stem}.txt"
        
        img_save_path = images_dir / image_name
        label_save_path = labels_dir / txt_name
        annotated_save_path = annotated_dir / f"{img_name_path.stem}_annotated.jpg"
        
        # Save original and label txt
        cv2.imwrite(str(img_save_path), img)
        
        # Generate YOLO lines
        lines = []
        for ann in annotations:
            label = ann.get("label", "person")
            cid = get_class_id(label, class_mapping)
            
            if ann["type"] == "bbox":
                # xc, yc, w, h (normalized)
                xc = ann["x"] + ann["w"] / 2.0
                yc = ann["y"] + ann["h"] / 2.0
                w = ann["w"]
                h = ann["h"]
                lines.append(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
            elif ann["type"] == "polygon":
                # x1 y1 x2 y2 ... (normalized)
                pts_str = " ".join([f"{p[0]:.6f} {p[1]:.6f}" for p in ann["points"]])
                lines.append(f"{cid} {pts_str}\n")
                
        with open(label_save_path, "w") as f:
            f.writelines(lines)
            
        # Draw and save annotated version
        ann_img = draw_annotations(img.copy(), annotations, class_mapping)
        cv2.imwrite(str(annotated_save_path), ann_img)
        
        # If user checked 'Save to raw dataset'
        if save_to_raw:
            raw_img_dir = dataset_root / "images"
            raw_lbl_dir = dataset_root / "labels"
            raw_img_dir.mkdir(parents=True, exist_ok=True)
            raw_lbl_dir.mkdir(parents=True, exist_ok=True)
            
            # Save YOLO txt format and image to raw dataset
            cv2.imwrite(str(raw_img_dir / image_name), img)
            with open(raw_lbl_dir / txt_name, "w") as f:
                f.writelines(lines)

    elif selected_format == "COCO":
        # COCO Text labels format: class_id x_min y_min width height (pixel values)
        coco_dir = base_dir / "coco"
        images_dir = coco_dir / "images"
        labels_dir = coco_dir / "labels"
        annotated_dir = coco_dir / "annotated"
        
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        annotated_dir.mkdir(parents=True, exist_ok=True)
        
        img_name_path = Path(image_name)
        txt_name = f"{img_name_path.stem}.txt"
        
        img_save_path = images_dir / image_name
        label_save_path = labels_dir / txt_name
        annotated_save_path = annotated_dir / f"{img_name_path.stem}_annotated.jpg"
        
        cv2.imwrite(str(img_save_path), img)
        
        lines = []
        for ann in annotations:
            label = ann.get("label", "person")
            cid = get_class_id(label, class_mapping)
            
            if ann["type"] == "bbox":
                x = int(ann["x"] * img_w)
                y = int(ann["y"] * img_h)
                w = int(ann["w"] * img_w)
                h = int(ann["h"] * img_h)
                lines.append(f"{cid} {x} {y} {w} {h}\n")
            elif ann["type"] == "polygon":
                # Convert points to pixel values
                pts_str = " ".join([f"{int(p[0]*img_w)} {int(p[1]*img_h)}" for p in ann["points"]])
                lines.append(f"{cid} {pts_str}\n")
                
        with open(label_save_path, "w") as f:
            f.writelines(lines)
            
        ann_img = draw_annotations(img.copy(), annotations, class_mapping)
        cv2.imwrite(str(annotated_save_path), ann_img)

    elif selected_format == "COCO JSON":
        # COCO JSON structure
        coco_json_dir = base_dir / "coco_json"
        images_dir = coco_json_dir / "images"
        ann_folder = coco_json_dir / "annotations"
        annotated_dir = coco_json_dir / "annotated"
        
        images_dir.mkdir(parents=True, exist_ok=True)
        ann_folder.mkdir(parents=True, exist_ok=True)
        annotated_dir.mkdir(parents=True, exist_ok=True)
        
        img_name_path = Path(image_name)
        json_path = ann_folder / "annotations.json"
        
        img_save_path = images_dir / image_name
        annotated_save_path = annotated_dir / f"{img_name_path.stem}_annotated.jpg"
        
        cv2.imwrite(str(img_save_path), img)
        
        # Read existing annotations.json or start new
        coco_data = {"images": [], "annotations": [], "categories": []}
        if json_path.exists():
            try:
                with open(json_path, "r") as f:
                    coco_data = json.load(f)
            except Exception:
                pass
                
        # Generate new IDs
        img_id = len(coco_data["images"]) + 1
        
        # Check if category ID mapping exists in categories list
        for ann in annotations:
            label = ann.get("label", "person")
            get_class_id(label, class_mapping)
            
        for name, cid in class_mapping.items():
            if not any(c["id"] == cid for c in coco_data["categories"]):
                coco_data["categories"].append({
                    "id": cid,
                    "name": name,
                    "supercategory": "none"
                })
                
        coco_data["images"].append({
            "id": img_id,
            "width": img_w,
            "height": img_h,
            "file_name": image_name
        })
        
        # Add annotations
        for ann in annotations:
            ann_id = len(coco_data["annotations"]) + 1
            label = ann.get("label", "person")
            cid = get_class_id(label, class_mapping)
            
            # Standard pixel values
            x = ann.get("x", 0) * img_w
            y = ann.get("y", 0) * img_h
            w = ann.get("w", 0) * img_w
            h = ann.get("h", 0) * img_h
            
            area = w * h
            segmentation = []
            
            if ann["type"] == "polygon":
                flat_pts = []
                for p in ann["points"]:
                    flat_pts.append(p[0] * img_w)
                    flat_pts.append(p[1] * img_h)
                segmentation = [flat_pts]
                
                # Calculate Shoelace Area for polygon
                n = len(ann["points"])
                poly_area = 0.0
                for i in range(n):
                    j = (i + 1) % n
                    px1 = ann["points"][i][0] * img_w
                    py1 = ann["points"][i][1] * img_h
                    px2 = ann["points"][j][0] * img_w
                    py2 = ann["points"][j][1] * img_h
                    poly_area += px1 * py2
                    poly_area -= px2 * py1
                area = abs(poly_area) / 2.0
                
                # Bounding box of the polygon
                xs = [p[0] * img_w for p in ann["points"]]
                ys = [p[1] * img_h for p in ann["points"]]
                x = min(xs)
                y = min(ys)
                w = max(xs) - x
                h = max(ys) - y
            else:
                # If bbox, build flat rectangle polygon segmentation list
                segmentation = [[x, y, x + w, y, x + w, y + h, x, y + h]]
                
            coco_data["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cid,
                "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                "segmentation": [[round(coord, 2) for coord in poly] for poly in segmentation],
                "area": round(area, 2),
                "iscrowd": 0
            })
            
        with open(json_path, "w") as f:
            json.dump(coco_data, f, indent=4)
            
        ann_img = draw_annotations(img.copy(), annotations, class_mapping)
        cv2.imwrite(str(annotated_save_path), ann_img)
        
    return True, f"Successfully saved to {selected_format} directory."
