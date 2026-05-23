# Real-time Object Counter

A computer vision project that detects and counts objects in real-time using YOLOv8 and OpenCV.

## What it does
- Detects objects live from webcam feed
- Draws bounding boxes around detected objects
- Counts total objects visible on screen in real time

## Tech stack
- Python
- YOLOv8 (Ultralytics)
- OpenCV

## How to run
1. Clone this repo
2. Install dependencies:
   pip install opencv-python ultralytics numpy
3. Run:
   python webcam_detect.py

> Note: YOLOv8 model downloads automatically on first run