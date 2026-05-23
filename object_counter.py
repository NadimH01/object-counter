import cv2
from ultralytics import YOLO
import csv
from datetime import datetime
import matplotlib.pyplot as plt

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Counting variables
count = 0
tracked_ids = set()
log_data = []  # stores [timestamp, count] for chart

# Create CSV file
csv_file = open("count_log.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["timestamp", "total_count", "object_label"])

print("Counter started! Press Q to quit. Press R to reset.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]
    line_y = height // 2
    cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 255), 2)

    results = model.track(frame, persist=True, verbose=False)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        classes = results[0].boxes.cls.cpu().numpy().astype(int)

        for box, obj_id, cls_id in zip(boxes, ids, classes):
            label = model.names[cls_id]
            x1, y1, x2, y2 = box
            center_y = int((y1 + y2) / 2)
            center_x = int((x1 + x2) / 2)

            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

            if abs(center_y - line_y) < 25 and obj_id not in tracked_ids:
                tracked_ids.add(obj_id)
                count += 1

                # Log to CSV
                timestamp = datetime.now().strftime("%H:%M:%S")
                csv_writer.writerow([timestamp, count, label])
                csv_file.flush()  # save immediately

                # Store for chart
                log_data.append((timestamp, count, label))
                print(f"[{timestamp}] Counted: {label} | Total: {count}")

    annotated_frame = results[0].plot(img=frame)

    cv2.putText(annotated_frame, f"Total Counted: {count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated_frame, "Counting Line", (10, line_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Object Counter", annotated_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('r'):
        count = 0
        tracked_ids = set()
        log_data = []
        print("Counter reset!")

cap.release()
cv2.destroyAllWindows()
csv_file.close()

print(f"\nSession ended. Total objects counted: {count}")
print("Saved to count_log.csv")

# Generate chart if anything was counted
if log_data:
    timestamps = [d[0] for d in log_data]
    counts = [d[1] for d in log_data]
    labels = [d[2] for d in log_data]

    plt.figure(figsize=(10, 5))
    plt.step(timestamps, counts, where="post", color="#1D9E75", linewidth=2)
    plt.scatter(timestamps, counts, color="#534AB7", zorder=5)

    # Label each point with object name
    for i, (t, c, l) in enumerate(zip(timestamps, counts, labels)):
        plt.annotate(l, (t, c), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)

    plt.title("Objects Counted Over Time")
    plt.xlabel("Time")
    plt.ylabel("Total Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("count_chart.png")
    plt.show()
    print("Chart saved as count_chart.png")
else:
    print("No objects counted — no chart generated.")