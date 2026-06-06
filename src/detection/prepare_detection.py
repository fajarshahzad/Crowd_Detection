import os
import shutil
import random
from pathlib import Path

def split_detection_dataset(
    raw_images_dir: str = "Dataset/images",
    raw_labels_dir: str = "Dataset/labels",
    output_dir: str = "Dataset/processed/detection",
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42
):
    """
    Splits detection images and labels into train/val/test splits.
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
        
    print(f"Found {len(all_images)} total raw images for detection split.")
    
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
                shutil.copy2(lbl_p, out_lbl_p)
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
        
    print(f"Detection dataset prepared at {output_dir}")
    return True

if __name__ == "__main__":
    split_detection_dataset()
