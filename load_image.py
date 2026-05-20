import cv2

# Load a real image from your folder
image = cv2.imread(r"C:\Project-counter\test.jpg")  # Make sure to have an image named 'test.jpg' in the same directory

# Print its info
print("Image shape:", image.shape)
print("Width:", image.shape[1])
print("Height:", image.shape[0])

# Display it
cv2.imshow("My Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()