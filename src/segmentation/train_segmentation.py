import os
from pathlib import Path
from ultralytics import YOLO

# Callback to output training progress in percentage
def on_train_epoch_end(trainer):
    epoch = trainer.epoch + 1
    epochs = trainer.epochs
    percentage = (epoch / epochs) * 100
    print(f"\n==========================================")
    print(f" TRAINING PROGRESS: {percentage:.1f}% ({epoch}/{epochs} Epochs Completed)")
    print(f"==========================================\n")

def train_segmentation(
    data_yaml: str = "Dataset/processed/segmentation/data.yaml",
    model_size: str = "yolov8n-seg.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    resume_checkpoint: str = None
):
    model_path = model_size
    should_resume = False

    # Check for resume options
    if resume_checkpoint and Path(resume_checkpoint).exists():
        print(f"Resuming segmentation training from specified checkpoint: {resume_checkpoint}")
        model_path = resume_checkpoint
        should_resume = True
    else:
        # Check standard YOLO segmentation output run folders for last.pt
        default_last = Path("runs/segment/train/weights/last.pt")
        if default_last.exists():
            print(f"Auto-detecting last checkpoint to resume: {default_last}")
            model_path = str(default_last)
            should_resume = True

    print(f"Loading Segmentation Model: {model_path} (resume={should_resume})")
    model = YOLO(model_path)
    
    # Add the custom progress callback
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    
    # Run training
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        resume=should_resume,
        device="cpu"  # Change to 0 or gpu if CUDA GPU is available
    )
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="YOLOv8 Segmentation Training Script")
    parser.add_argument("--data", type=str, default="Dataset/processed/segmentation/data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n-seg.pt", help="Base model size or checkpoint")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--resume", type=str, default=None, help="Path to specific checkpoint")
    
    args = parser.parse_args()
    train_segmentation(
        data_yaml=args.data,
        model_size=args.model,
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch,
        resume_checkpoint=args.resume
    )
