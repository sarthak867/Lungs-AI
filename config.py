import os
from pathlib import Path

APP_NAME = "LungLens AI"
APP_SUBTITLE = "Explainable Chest X-Ray Analysis"
MODEL_NAME = "TorchXRayVision DenseNet-121"
CHECKPOINT_ID = "densenet121-res224-all"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
MODEL_INPUT_SIZE = 224
PROJECT_ROOT = Path(__file__).resolve().parent
XRAY_CLASSIFIER_CHECKPOINT = PROJECT_ROOT / "xray_classifier.pth"
XRAY_CLASSIFIER_ID = "local-mobilenetv2-xray-classifier-v1"
XRAY_CLASSIFIER_THRESHOLD = float(os.environ.get("LUNGLENS_XRAY_THRESHOLD", "0.8"))

if not 0.0 <= XRAY_CLASSIFIER_THRESHOLD <= 1.0:
    raise ValueError("LUNGLENS_XRAY_THRESHOLD must be between 0 and 1.")
