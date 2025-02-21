import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load the model
model = YOLO(r'models\license_plate_detector.pt')

# Function to perform license plate detection
def detect_license_plate(image_path):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image {image_path}")
        return

    # Perform inference
    results = model(img)

    # Plot the results
    for result in results:
        annotated_img = result.plot()
        plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

# Example usage
#detect_license_plate(r"C:\Users\jacqu\OneDrive\Documents\Insane deadlines\CNN\raw\2.png")
