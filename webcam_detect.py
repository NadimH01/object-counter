import cv2
from ultralytics import YOLO

# Use a larger model for better accuracy than yolov8n
model = YOLO("yolov8s.pt")

# CAP_DSHOW fixes webcam detection on Windows
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("Webcam started! Press Q to quit.")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Can't access webcam")
        break

    # Increase confidence threshold and keep a stable IoU for cleaner detections
    results = model(frame, conf=0.4, iou=0.5, verbose=False)
    count = len(results[0].boxes)
    annotated_frame = results[0].plot()

    cv2.putText(annotated_frame, f"Objects: {count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Live Detection", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()