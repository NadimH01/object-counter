import cv2
import numpy as np

# Load an image (we'll create a simple one since you may not have one ready)
image = np.zeros((400, 600, 3), dtype=np.uint8)

# Draw some shapes so we have something to look at
cv2.rectangle(image, (100, 100), (300, 300), (0, 255, 0), 3)
cv2.circle(image, (450, 200), 80, (0, 0, 255), -1)
cv2.putText(image, "Hello OpenCV!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Print the image info
print("Image shape:", image.shape)
print("Width:", image.shape[1], "Height:", image.shape[0])

# Display it
print("Image shape:", image.shape)
print("First pixel value:", image[0, 0])
print("Green rectangle pixel:", image[150, 150])
print("Blue circle pixel:", image[200, 450])
cv2.imshow("My First Image", image)
cv2.waitKey(0)  # Wait until you press any key
cv2.destroyAllWindows()