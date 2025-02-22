from flask import Flask, render_template, request, send_file
import cv2
import numpy as np
from ultralytics import YOLO
import os

# Initialize Flask
app = Flask(__name__)

# Load YOLO model
model = YOLO('CNN\\models\license_plate_detector.pt')

# Define directories
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'

# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if file has an allowed image extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            print("❌ No file part in request.")
            return "No file uploaded", 400
        
        file = request.files['file']
        if file.filename == '':
            print("❌ No selected file.")
            return "No file selected", 400

        if not allowed_file(file.filename):
            print(f"❌ Unsupported file type: {file.filename}")
            return "Unsupported file type", 400

        # Save uploaded file
        filename = os.path.join(UPLOAD_FOLDER, file.filename)
        filename = os.path.abspath(filename)  # Absolute path for safety
        file.save(filename)
        print(f"✅ File saved: {filename}")

        # Process the image
        result_path = detect_license_plate(filename)

        # Check if processing was successful
        if result_path and os.path.exists(result_path):
            print(f"✅ Processed image ready: {result_path}")
            return send_file(result_path, mimetype='image/jpeg')
        else:
            print(f"❌ Error processing image: {filename}")
            return "Error processing image", 500

    return render_template('index.html')

def detect_license_plate(image_path):
    print(f"🔍 Loading image: {image_path}")
    img = cv2.imread(image_path)

    if img is None:
        print(f"❌ Error: Unable to load image {image_path}")
        return None

    results = model(img)

    for result in results:
        annotated_img = result.plot()
        result_filename = os.path.join(RESULT_FOLDER, os.path.basename(image_path))
        result_filename = os.path.abspath(result_filename)  # Ensure absolute path

        # Save the annotated image
        success = cv2.imwrite(result_filename, annotated_img)

        if success:
            print(f"✅ Saved result: {result_filename}")
            return result_filename
        else:
            print(f"❌ Failed to save result: {result_filename}")
            return None

if __name__ == '__main__':
    print("🚀 Flask app running...")
    app.run(debug=True)
