from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from config import (
    XRAY_CLASSIFIER_CHECKPOINT,
    XRAY_CLASSIFIER_ID,
    XRAY_CLASSIFIER_THRESHOLD,
)
from services.modality_validation import SuitabilityDecision, SuitabilityServiceUnavailable, SuitabilityStatus

logger = logging.getLogger(__name__)
class XRayClassifierError(RuntimeError):
    """Raised when the local X-ray-vs-photo model fails during inference."""


@lru_cache(maxsize=1)
def _load_model(checkpoint_path: str, modified_ns: int, file_size: int):
    del modified_ns, file_size
    path = Path(checkpoint_path)
    if not path.is_file():
        raise SuitabilityServiceUnavailable(
            f"Chest X-ray validation model not found at the configured project checkpoint ({path.name})."
        )
    try:
        import torch
        from torchvision import models

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            raise ValueError("Checkpoint must be a dictionary containing state_dict and classes.")
        state_dict = checkpoint["state_dict"]
        classes = checkpoint["classes"]
        _validate_class_order(classes)
        model = models.mobilenet_v2(weights=None)
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(model.last_channel, 2),
        )
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model, torch, classes
    except SuitabilityServiceUnavailable:
        raise
    except Exception as exc:
        logger.exception("Could not load the local X-ray suitability checkpoint.")
        raise SuitabilityServiceUnavailable(
            "The local MobileNetV2 X-ray validation model could not be loaded. Verify xray_classifier.pth contains matching state_dict weights and classes metadata."
        ) from exc


def _validate_class_order(classes) -> None:
    if not isinstance(classes, (list, tuple)) or len(classes) != 2:
        raise ValueError("Checkpoint classes must contain exactly two class labels.")
    if "xray" not in classes or "non_xray" not in classes:
        raise ValueError("Checkpoint classes must include non_xray and xray.")


def _checkpoint_signature() -> tuple[str, int, int]:
    path = XRAY_CLASSIFIER_CHECKPOINT
    try:
        stats = path.stat()
    except OSError:
        return str(path), 0, 0
    return str(path), stats.st_mtime_ns, stats.st_size


def classifier_revision() -> str:
    _, modified_ns, file_size = _checkpoint_signature()
    return f"{XRAY_CLASSIFIER_ID}:{modified_ns}:{file_size}:{XRAY_CLASSIFIER_THRESHOLD:.12g}"


def preprocess_for_classifier(image: Image.Image):
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return transform(image.convert("RGB")).unsqueeze(0)


def predict_xray_score(image: Image.Image) -> float:
    path, modified_ns, file_size = _checkpoint_signature()
    model, torch, classes = _load_model(path, modified_ns, file_size)
    try:
        tensor = preprocess_for_classifier(image)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model = model.to(device)
            tensor = tensor.to(device)
            with torch.no_grad():
                logits = model(tensor)
        except Exception:
            if device.type != "cuda":
                raise
            logger.exception("X-ray classifier GPU inference failed; retrying on CPU.")
            device = torch.device("cpu")
            model = model.to(device)
            tensor = tensor.to(device)
            with torch.no_grad():
                logits = model(tensor)

        if logits.ndim != 2 or logits.shape != (1, len(classes)):
            raise XRayClassifierError("The X-ray classifier returned an unexpected output shape.")
        probabilities = torch.softmax(logits, dim=1)[0]
        values = probabilities.detach().cpu().numpy()
        if not np.isfinite(values).all():
            raise XRayClassifierError("The X-ray classifier returned non-finite scores.")
        return float(values[classes.index("xray")])
    except XRayClassifierError:
        raise
    except Exception as exc:
        logger.exception("Local X-ray-vs-photo classifier inference failed.")
        raise XRayClassifierError("The local image-validation model could not classify this image.") from exc


def classify_chest_xray(image: Image.Image) -> SuitabilityDecision:
    try:
        xray_score = predict_xray_score(image)
    except SuitabilityServiceUnavailable:
        raise
    except XRayClassifierError:
        raise
    if xray_score >= XRAY_CLASSIFIER_THRESHOLD:
        return SuitabilityDecision(
            status=SuitabilityStatus.ACCEPTED,
            reason="The local classifier selected the X-Ray class.",
            xray_score=xray_score,
            classifier_id=XRAY_CLASSIFIER_ID,
        )
    return SuitabilityDecision(
        status=SuitabilityStatus.REJECTED,
        reason="The local classifier selected the non-X-ray class or scored below the configured threshold.",
        xray_score=xray_score,
        classifier_id=XRAY_CLASSIFIER_ID,
    )
