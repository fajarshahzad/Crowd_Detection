import cv2
import os
from pathlib import Path

class BBoxAnnotator:
    def __init__(self, images_dir=None, labels_dir=None, class_name="person"):
        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        self.images_dir = Path(images_dir) if images_dir else PROJECT_ROOT / "Dataset" / "images"
        self.labels_dir = Path(labels_dir) if labels_dir else PROJECT_ROOT / "Dataset" / "labels"
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        
        self.class_name = class_name
        self.class_id = 0
        
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        self.image_paths = sorted([p for p in self.images_dir.iterdir() if p.suffix.lower() in valid_exts]) if self.images_dir.exists() else []
        self.img_index = 0
        
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.rx, self.ry = -1, -1
        self.bboxes = []
        
    def load_annotations(self, image_path):
        self.bboxes = []
        txt_path = self.labels_dir / f"{image_path.stem}.txt"
        if txt_path.exists():
            with open(txt_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cid = int(parts[0])
                        x_c = float(parts[1])
                        y_c = float(parts[2])
                        w = float(parts[3])
                        h = float(parts[4])
                        self.bboxes.append([cid, x_c, y_c, w, h])
                        
    def save_annotations(self, image_path):
        txt_path = self.labels_dir / f"{image_path.stem}.txt"
        with open(txt_path, "w") as f:
            for bbox in self.bboxes:
                f.write(f"{bbox[0]} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {bbox[4]:.6f}\n")
                
    def draw_callback(self, event, x, y, flags, param):
        img_h, img_w = param[0], param[1]
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
            self.rx, self.ry = x, y
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.rx, self.ry = x, y
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            x1, y1 = min(self.ix, x), min(self.iy, y)
            x2, y2 = max(self.ix, x), max(self.iy, y)
            
            box_w = x2 - x1
            box_h = y2 - y1
            
            if box_w > 5 and box_h > 5:
                x_center = (x1 + box_w / 2.0) / img_w
                y_center = (y1 + box_h / 2.0) / img_h
                w_norm = box_w / img_w
                h_norm = box_h / img_h
                self.bboxes.append([self.class_id, x_center, y_center, w_norm, h_norm])
                
    def run(self):
        if not self.image_paths:
            print("No images found to annotate.")
            return
            
        cv2.namedWindow("Annotator", cv2.WINDOW_NORMAL)
        
        while self.img_index < len(self.image_paths):
            img_path = self.image_paths[self.img_index]
            img = cv2.imread(str(img_path))
            if img is None:
                self.img_index += 1
                continue
                
            self.load_annotations(img_path)
            h, w, _ = img.shape
            
            cv2.setMouseCallback("Annotator", self.draw_callback, [h, w])
            
            while True:
                temp_img = img.copy()
                
                for bbox in self.bboxes:
                    _, xc, yc, wn, hn = bbox
                    x1 = int((xc - wn/2.0) * w)
                    y1 = int((yc - hn/2.0) * h)
                    x2 = int((xc + wn/2.0) * w)
                    y2 = int((yc + hn/2.0) * h)
                    cv2.rectangle(temp_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(temp_img, self.class_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                if self.drawing:
                    cv2.rectangle(temp_img, (self.ix, self.iy), (self.rx, self.ry), (0, 0, 255), 2)
                    
                info_text = f"Image {self.img_index + 1}/{len(self.image_paths)}: {img_path.name} | Press 'c' to clear, 's' to save/next, 'a' to prev, 'q' to quit"
                cv2.setWindowTitle("Annotator", info_text)
                cv2.imshow("Annotator", temp_img)
                
                key = cv2.waitKey(30) & 0xFF
                if key == ord('s'):
                    self.save_annotations(img_path)
                    self.img_index += 1
                    break
                elif key == ord('a'):
                    self.save_annotations(img_path)
                    self.img_index = max(0, self.img_index - 1)
                    break
                elif key == ord('c'):
                    self.bboxes = []
                elif key == ord('q'):
                    self.save_annotations(img_path)
                    cv2.destroyAllWindows()
                    return
                    
        cv2.destroyAllWindows()

if __name__ == "__main__":
    annotator = BBoxAnnotator()
    annotator.run()
