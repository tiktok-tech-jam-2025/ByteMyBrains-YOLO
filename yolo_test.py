import cv2
from ultralytics import YOLO
import os
import glob

# --- Load custom YOLOv12 model ---
model = YOLO("openimage.pt")  # change to your .pt
print(model.names)

# Class names dictionary
class_names = model.names  # {0: 'person', 1: 'cat', ...}

# Input and output directories
input_dir = r"test"
output_dir = r"output_openimage"
os.makedirs(output_dir, exist_ok=True)

# Get all images in the folder (jpg, png, jpeg)
image_paths = glob.glob(os.path.join(input_dir, "*.*"))
image_paths = [p for p in image_paths if p.lower().endswith(('.jpg', '.jpeg', '.png'))]

for image_path in image_paths:
    print(f"Processing {image_path} ...")
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load {image_path}, skipping...")
        continue
    
    # Run YOLO inference
    results = model.predict(source=img, conf=0.25)
    
    # Process results
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            class_id = int(classes[i])
            class_name = class_names[class_id]
            conf = scores[i]

            label = f"{class_name}:{conf:.2f}"

            # Draw bounding box
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Text settings
            font_scale = 0.7
            font_thickness = 2
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Get size of text
            (w, h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)

            # Draw filled rectangle for text background
            cv2.rectangle(img, (int(x1), int(y1)-h-5), (int(x1)+w, int(y1)), (0, 255, 0), -1)

            # Put text (white color on green background)
            cv2.putText(img, label, (int(x1), int(y1)-5), font, font_scale, (255, 255, 255), font_thickness)

            # Print detection info
            print(f"{os.path.basename(image_path)} -> {class_name} ({class_id}), "
                f"Conf: {conf:.2f}, Box: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")

    
    # Save annotated image
    save_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(save_path, img)
    print(f"Saved annotated image to {save_path}\n")
