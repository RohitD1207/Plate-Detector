from ultralytics import YOLO

# Load YOLOv8 model (Nano version for speed)
model = YOLO("yolov8n.pt")

# Train on your dataset
model.train(data="dataset.yaml", epochs=50, batch=16, imgsz=640, workers=0)

model.save('license-plate-detector')  # Export to ONNX for deployment