# License Plate Detection using YOLO & Flask

## Description

This is a Flask-based web application for license plate detection using a YOLO-based model. Users can upload images, and the model processes them to highlight detected license plates.

## Features

- Accepts common image formats (JPG, PNG, BMP, etc.)
- Uses YOLO for license plate detection
- Flask-powered web interface
- Saves processed images

## Installation

### Clone the repository

```bash
git clone https://github.com/RohitD1207/Plate-Detector.git
cd REPO_NAME
```

### Set up a virtual environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the app

```bash
python app.py
```

Now visit `http://127.0.0.1:5000/` in your browser.

## Project Structure

```
📁 flask_app
 ├── 📁 models/                # Trained YOLO model
 ├── 📁 uploads/               # Uploaded images
 ├── 📁 results/               # Processed images with detections
 ├── app.py                    # Main Flask application
 ├── requirements.txt           # Dependencies
 ├── README.md                  # Project documentation
 └── templates/
      └── index.html            # Web interface
```

## Dependencies

- Flask
- OpenCV
- NumPy
- Ultralytics (YOLO)

Install them with:

```bash
pip install flask opencv-python numpy ultralytics
```

## License

MIT

## Contributing

Fork the repository and submit a pull request if you make improvements.

## Contact

For issues, open a GitHub Issue or reach out privately.

