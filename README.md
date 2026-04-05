# Face Mask Detector (Flask + Keras CNN)

A web app that predicts whether a face is wearing a mask from an uploaded image.

## Features
- Train a CNN model from local image dataset folders.
- Flask web UI for image upload and prediction.
- Face-aware inference (automatic face crop when OpenCV is available).
- Test-time augmentation at inference (original + flipped image averaging).
- Confidence gate (`uncertain` below threshold).

## Tech Stack
- Frontend: HTML, CSS, JavaScript
- Backend: Python, Flask
- Model: TensorFlow/Keras CNN
- Inference utils: Pillow, OpenCV (headless)

## Project Structure
```text
face-mask-detector/
|- app.py
|- train_model.py
|- requirements.txt
|- Procfile
|- templates/
|  |- index.html
|- dataset/
|  |- with_mask/
|  |- without_mask/
|- model/
|  |- mask_detector.h5
|  |- class_names.json
|  |- training_config.json
```

## Dataset Format
Place your dataset like this:

```text
dataset/
|- with_mask/
|  |- image1.jpg
|  |- image2.jpg
|- without_mask/
   |- image1.jpg
   |- image2.jpg
```

Supported extensions: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`.

## Local Setup (Windows PowerShell)
1. Clone and open the project:
```powershell
git clone <your-repo-url>
cd face-mask-detector
```

2. Create and activate virtual environment:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. Install dependencies:
```powershell
pip install -r requirements.txt
```

## Train the Model
Default training:
```powershell
python train_model.py
```

Recommended training (better accuracy):
```powershell
python train_model.py --epochs 10 --img-size 128 --batch-size 32
```

Quick smoke test:
```powershell
python train_model.py --epochs 1 --img-size 64 --batch-size 256
```

Training outputs are written to `model/`:
- `mask_detector.h5`
- `class_names.json`
- `training_config.json`

## Run the Web App (Localhost)
Start server:
```powershell
python app.py
```

Open in browser:
- `http://127.0.0.1:5000`

Upload an image and click **Analyze**.

## API Endpoint
`POST /predict`

Form-data field:
- `image`: image file (`png`, `jpg`, `jpeg`, `webp`)

Example success response:
```json
{
  "prediction": "with_mask",
  "confidence": 0.97,
  "probabilities": {
    "with_mask": 0.97,
    "without_mask": 0.03
  }
}
```

## Notes on Accuracy
If predictions are inconsistent:
- Retrain with more epochs.
- Ensure dataset quality and balanced classes.
- Use clear frontal-face images for best results.
- Restart `app.py` after retraining so latest model is loaded.

## Deployment
`Procfile` is included for WSGI hosting:
```text
web: gunicorn app:app
```

## Troubleshooting
- `Model not found`:
  - Run `python train_model.py` first.
- `Address already in use`:
  - Use a different port:
  ```powershell
  $env:PORT=5001
  python app.py
  ```
- Pylance unresolved imports:
  - Select project interpreter: `venv\Scripts\python.exe`
  - Restart Pylance/VS Code window.

## Git Hygiene (Recommended)
Do not commit:
- `venv/`
- `dataset/` (raw images)
- `model/*.h5` and large generated artifacts
- `.env` and secrets

Use placeholders/docs so others can reproduce safely.
