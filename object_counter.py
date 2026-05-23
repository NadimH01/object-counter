import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Counting variables
count = 0
tracked_ids = set()

print("Counter started! Press Q to quit. Press R to reset counter.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    # Counting line near the top
    line_y = height // 2
    cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 255), 2)

    # Run detection with tracking
    results = model.track(frame, persist=True, verbose=False)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, obj_id in zip(boxes, ids):
            x1, y1, x2, y2 = box
            center_y = int((y1 + y2) / 2)
            center_x = int((x1 + x2) / 2)

            # Draw center dot
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

            # Count if crosses the line
            if abs(center_y - line_y) < 25 and obj_id not in tracked_ids:
                tracked_ids.add(obj_id)
                count += 1

    # Draw annotations
    annotated_frame = results[0].plot(img=frame)

    # Show count on screen
    cv2.putText(annotated_frame, f"Total Counted: {count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated_frame, "Counting Line", (10, line_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Object Counter", annotated_frame)

    # Key controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('r'):
        count = 0
        tracked_ids = set()
        print("Counter reset!")

cap.release()
cv2.destroyAllWindows()
print(f"Session ended. Total objects counted: {count}")