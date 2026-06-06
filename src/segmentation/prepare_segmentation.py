import os
import shutil
import random
from pathlib import Path

def split_segmentation_dataset(
    raw_images_dir: str = "Dataset/images",
    raw_labels_dir: str = "Dataset/labels",
    output_dir: str = "Dataset/processed/segmentation",
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42
):
    """
    Splits segmentation images and labels into train/val/test splits.
    """
    random.seed(seed)
    
    images_src = Path(raw_images_dir)
    labels_src = Path(raw_labels_dir)
    out_path = Path(output_dir)
    
    if not images_src.exists():
        print(f"Error: Images directory not found at {images_src}")
        return False
        
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    all_images = [p for p in images_src.iterdir() if p.suffix.lower() in valid_exts]
    
    if not all_images:
        print(f"Error: No images found in {images_src}")
        return False
        
    print(f"Found {len(all_images)} total raw images for segmentation split.")
    
    random.shuffle(all_images)
    num_imgs = len(all_images)
    
    train_end = int(num_imgs * train_ratio)
    val_end = train_end + int(num_imgs * val_ratio)
    
    splits = {
        "train": all_images[:train_end],
        "val": all_images[train_end:val_end],
        "test": all_images[val_end:]
    }
    
    for split in ["train", "val", "test"]:
        (out_path / split / "images").mkdir(parents=True, exist_ok=True)
        (out_path / split / "labels").mkdir(parents=True, exist_ok=True)
        
    for split_name, img_list in splits.items():
        print(f"Processing split '{split_name}' with {len(img_list)} images...")
        for img_p in img_list:
            shutil.copy2(img_p, out_path / split_name / "images" / img_p.name)
            
            lbl_p = labels_src / f"{img_p.stem}.txt"
            out_lbl_p = out_path / split_name / "labels" / f"{img_p.stem}.txt"
            if lbl_p.exists():
                # For segmentation training, if user provided standard YOLO bboxes, we can auto-convert 
                # them to a 4-point polygon representation for YOLO-seg training.
                # Format: class_id x1 y1 x2 y2 x3 y3 x4 y4
                with open(lbl_p, "r") as f_in:
                    lines = f_in.readlines()
                with open(out_lbl_p, "w") as f_out:
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cid, xc, yc, w, h = map(float, parts)
                            # Convert center-width-height to bounding polygon (4 vertices)
                            x1 = xc - w / 2.0
                            y1 = yc - h / 2.0
                            x2 = xc + w / 2.0
                            y2 = yc - h / 2.0
                            x3 = xc + w / 2.0
                            y3 = yc + h / 2.0
                            x4 = xc - w / 2.0
                            y4 = yc + h / 2.0
                            f_out.write(f"{int(cid)} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f}\n")
            else:
                open(out_lbl_p, "w").close()
                
    # Create data.yaml
    yaml_content = f"""path: {out_path.resolve().as_posix()}
train: train/images
val: val/images
test: test/images

# Classes
names:
  0: person
"""
    yaml_path = out_path / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
        
    print(f"Segmentation dataset prepared at {output_dir}")
    return True

if __name__ == "__main__":
    split_segmentation_dataset()
