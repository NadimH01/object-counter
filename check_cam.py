import cv2

# Try camera indexes 0, 1, 2
for i in range(3):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera found at index {i}")
        cap.release()
    else:
        print(f"No camera at index {i}")