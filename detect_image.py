import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model("test1.jpeg")

# Show the image with bounding boxes
results[0].show()

# Print details for each object
print(f"Total objects detected: {len(results[0].boxes)}")
print("---")

for i, box in enumerate(results[0].boxes):
    class_id = int(box.cls)
    confidence = float(box.conf)
    label = model.names[class_id]
    
    x1, y1, x2, y2 = box.xyxy[0]
    width = int(x2 - x1)
    height = int(y2 - y1)
    
    print(f"Object {i+1}: {label}")
    print(f"  Confidence : {confidence:.2f}")
    print(f"  Box size   : {width}x{height} pixels")