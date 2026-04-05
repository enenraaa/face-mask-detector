import json
import os
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request
from keras.models import load_model
from PIL import Image

try:
    import cv2
except Exception:
    cv2 = None

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_H5_PATH = MODEL_DIR / "mask_detector.h5"
MODEL_KERAS_PATH = MODEL_DIR / "mask_detector.keras"
LABELS_PATH = MODEL_DIR / "class_names.json"
CONFIG_PATH = MODEL_DIR / "training_config.json"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
CONFIDENCE_THRESHOLD = 0.60

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8MB upload limit


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_class_names() -> list[str]:
    if LABELS_PATH.exists():
        with LABELS_PATH.open("r", encoding="utf-8") as f:
            labels = json.load(f)
            if isinstance(labels, list) and labels:
                return labels
    return ["with_mask", "without_mask"]


def load_image_size() -> int:
    if CONFIG_PATH.exists():
        try:
            with CONFIG_PATH.open("r", encoding="utf-8") as f:
                cfg = json.load(f)
            img_size = int(cfg.get("img_size", 128))
            if img_size > 0:
                return img_size
        except Exception:
            pass
    return 128


def preprocess_image(uploaded_file) -> np.ndarray:
    image = Image.open(uploaded_file.stream).convert("RGB")
    image_array = np.array(image, dtype=np.uint8)

    if cv2 is not None:
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            face_detector = cv2.CascadeClassifier(cascade_path)
            if not face_detector.empty():
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
                faces = face_detector.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(40, 40),
                )
                if len(faces) > 0:
                    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
                    pad = int(0.2 * max(w, h))
                    x1 = max(0, x - pad)
                    y1 = max(0, y - pad)
                    x2 = min(image_array.shape[1], x + w + pad)
                    y2 = min(image_array.shape[0], y + h + pad)
                    image_array = image_array[y1:y2, x1:x2]
        except Exception:
            pass

    image = Image.fromarray(image_array).resize((IMG_SIZE, IMG_SIZE))
    image_array = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)


CLASS_NAMES = load_class_names()
IMG_SIZE = load_image_size()
if MODEL_H5_PATH.exists():
    MODEL = load_model(MODEL_H5_PATH)
elif MODEL_KERAS_PATH.exists():
    MODEL = load_model(MODEL_KERAS_PATH)
else:
    MODEL = None


@app.route("/")
def index():
    return render_template("index.html")


@app.post("/predict")
def predict():
    if MODEL is None:
        return (
            jsonify(
                {
                    "error": "Model not found. Train first with: python train_model.py",
                }
            ),
            500,
        )

    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type."}), 400

    try:
        image_batch = preprocess_image(file)
        flipped_batch = np.flip(image_batch, axis=2)
        probabilities = (
            MODEL.predict(image_batch, verbose=0)[0]
            + MODEL.predict(flipped_batch, verbose=0)[0]
        ) / 2.0
        predicted_idx = int(np.argmax(probabilities))
        predicted_label = (
            CLASS_NAMES[predicted_idx]
            if predicted_idx < len(CLASS_NAMES)
            else f"class_{predicted_idx}"
        )
        confidence = float(probabilities[predicted_idx])

        if confidence < CONFIDENCE_THRESHOLD:
            predicted_label = "uncertain"

        class_probabilities = {}
        for idx, prob in enumerate(probabilities):
            name = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"class_{idx}"
            class_probabilities[name] = float(prob)

        return jsonify(
            {
                "prediction": predicted_label,
                "confidence": confidence,
                "probabilities": class_probabilities,
            }
        )
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "1") == "1",
    )
