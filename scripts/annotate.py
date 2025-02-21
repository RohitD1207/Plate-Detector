import os
import cv2
import glob
import torch
from ultralytics import YOLO
from PIL import Image
from tkinter import Tk
from tkinter.filedialog import askdirectory

# Select dataset directory
Tk().withdraw()  # Hide Tkinter root window
data_dir = askdirectory(title="Select Dataset Folder")
output_dir = os.path.join(data_dir, "converted")
labels_dir = os.path.join(data_dir, "labels")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

# Convert all images to .jpg format
image_paths = glob.glob(os.path.join(data_dir, '*.*'))  # Get all files
for img_path in image_paths:
    try:
        img = Image.open(img_path)
        new_path = os.path.join(output_dir, os.path.splitext(os.path.basename(img_path))[0] + ".jpg")
        img.convert('RGB').save(new_path, "JPEG")
    except Exception as e:
        print(f"Error converting {img_path}: {e}")

print("All images converted to .jpg and saved in 'converted' folder.")

# Load pretrained YOLO model for auto-labeling
yolo_model = YOLO("yolov8n.pt")  # Using YOLOv8 nano for speed

# Auto-label images
def detect_and_save_labels(image_path, output_label_dir):
    img = cv2.imread(image_path)
    results = yolo_model(img)
    h, w, _ = img.shape
    label_path = os.path.join(output_label_dir, os.path.splitext(os.path.basename(image_path))[0] + ".txt")
    
    with open(label_path, "w") as f:
        for result in results:
            for box in result.boxes:
                x_center = (box.xywh[0][0] / w).item()
                y_center = (box.xywh[0][1] / h).item()
                width = (box.xywh[0][2] / w).item()
                height = (box.xywh[0][3] / h).item()
                f.write(f"0 {x_center} {y_center} {width} {height}\n")

# Process all images
for img_path in glob.glob(os.path.join(output_dir, "*.jpg")):
    detect_and_save_labels(img_path, labels_dir)

print("✅ Auto-labeling complete. Labels saved in 'labels' folder.")
