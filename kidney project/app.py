from flask import Flask, render_template, request
from ultralytics import YOLO
import os
import uuid
import shutil

app = Flask(__name__)

# Load YOLOv8 model
model = YOLO("best.pt")  # Replace with your own model path if needed

# Ensure folders exist
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return "❌ No file part in request"

    file = request.files["file"]
    if file.filename == "":
        return "❌ No selected file"

    # Save uploaded (raw) image
    img_name = str(uuid.uuid4()) + ".jpg"
    img_path = os.path.join(UPLOAD_FOLDER, img_name)
    file.save(img_path)

    # Remove previous YOLO runs
    shutil.rmtree("runs/detect", ignore_errors=True)

    # Run YOLOv8 prediction
    results = model.predict(img_path, save=True)
    result = results[0]

    # Check for stone detection
    stone_present = len(result.boxes) > 0

    # Locate predicted image from YOLO
    latest_folder = sorted(os.listdir("runs/detect"))[-1]
    prediction_path = os.path.join("runs/detect", latest_folder)
    result_file = [f for f in os.listdir(prediction_path) if f.endswith((".jpg", ".png"))][0]

    # Copy predicted image to static/results
    result_name = str(uuid.uuid4()) + ".jpg"
    result_img_path = os.path.join(RESULT_FOLDER, result_name)
    shutil.copy(os.path.join(prediction_path, result_file), result_img_path)

    return render_template("index.html",
                           raw_img=os.path.join("static/uploads", img_name),
                           result_img=os.path.join("static/results", result_name),
                           stone_present=stone_present)

if __name__ == "__main__":
    app.run(debug=True)
