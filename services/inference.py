from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image

from config import (
    DISEASE_MODEL_CHECKPOINT,
    DISEASE_THRESHOLDS_FILE,
    MODEL_INPUT_SIZE,
    MODEL_NAME,
)
from services.modality_validation import (
    SuitabilityValidationResult,
    fingerprint_image,
    require_suitability_pass,
)

logger = logging.getLogger(__name__)

PATHOLOGY_LABELS = (
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
)


class ModelUnavailableError(RuntimeError):
    """Raised when the local disease model cannot be initialized."""


class InferenceError(RuntimeError):
    """Raised when a loaded model cannot produce a valid result."""


@dataclass(frozen=True)
class Prediction:
    label: str
    score: float
    threshold: float | None = None

    @property
    def meets_threshold(self) -> bool:
        return self.threshold is not None and self.score >= self.threshold


@dataclass(frozen=True)
class AnalysisResult:
    model_name: str
    checkpoint_id: str
    timestamp: str
    predictions: tuple[Prediction, ...]
    score_semantics: str
    raw_scores: tuple[float, ...]
    pathologies: tuple[str, ...]
    image_hash: str = ""
    image_fingerprint: str = ""
    model_revision: str = ""

    @property
    def findings(self) -> tuple[Prediction, ...]:
        return tuple(
            sorted(
                (prediction for prediction in self.predictions if prediction.meets_threshold),
                key=lambda prediction: prediction.score,
                reverse=True,
            )
        )


def _file_signature(path: Path) -> tuple[str, int, int]:
    try:
        stats = path.stat()
    except OSError:
        return str(path), 0, 0
    return str(path), stats.st_mtime_ns, stats.st_size


def disease_model_revision() -> str:
    return ":".join(
        str(value)
        for value in (*_file_signature(DISEASE_MODEL_CHECKPOINT), *_file_signature(DISEASE_THRESHOLDS_FILE))
    )


@lru_cache(maxsize=1)
def _load_model(checkpoint_path: str, modified_ns: int, file_size: int):
    del modified_ns, file_size
    path = Path(checkpoint_path)
    if not path.is_file():
        raise ModelUnavailableError(f"Disease model checkpoint not found: {path.name}.")
    try:
        import torch
        from torchvision import models

        model = models.densenet121(weights=None)
        model.classifier = torch.nn.Linear(model.classifier.in_features, len(PATHOLOGY_LABELS))
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state_dict, Mapping) or not state_dict:
            raise ValueError("Checkpoint must contain a non-empty raw state dictionary.")
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model, torch
    except Exception as exc:
        logger.exception("Could not load the project DenseNet-121 disease checkpoint.")
        raise ModelUnavailableError(
            "The local disease model could not be loaded. Verify the DenseNet-121 checkpoint and PyTorch installation."
        ) from exc


def load_model():
    path, modified_ns, file_size = _file_signature(DISEASE_MODEL_CHECKPOINT)
    return _load_model(path, modified_ns, file_size)


def load_thresholds() -> dict[str, float]:
    path = DISEASE_THRESHOLDS_FILE
    if not path.is_file():
        raise ModelUnavailableError(f"Disease threshold file not found: {path.name}.")
    try:
        raw_thresholds = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw_thresholds, dict) or set(raw_thresholds) != set(PATHOLOGY_LABELS):
            raise ValueError("Threshold file labels do not match the model output labels.")
        thresholds: dict[str, float] = {}
        for label in PATHOLOGY_LABELS:
            entry = raw_thresholds[label]
            threshold = entry["threshold"] if isinstance(entry, dict) else entry
            value = float(threshold)
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"Invalid threshold for {label}.")
            thresholds[label] = value
        return thresholds
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        logger.exception("Could not read the per-disease threshold file.")
        raise ModelUnavailableError(
            "The per-disease threshold file is invalid. Verify Disease_model/thresholds_my.json."
        ) from exc


def preprocess_disease_image(image: Image.Image):
    import torch
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    grayscale_rgb = image.convert("L").resize(
        (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
        Image.Resampling.BILINEAR,
    ).convert("RGB")
    return transform(grayscale_rgb).unsqueeze(0)


def run_inference(
    image: Image.Image,
    *,
    image_hash: str,
    validation: SuitabilityValidationResult | None,
) -> AnalysisResult:
    from datetime import datetime, timezone

    require_suitability_pass(validation, image_hash, image)
    model, torch = load_model()
    thresholds = load_thresholds()
    model_revision = disease_model_revision()
    try:
        tensor = preprocess_disease_image(image)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model = model.to(device)
            tensor = tensor.to(device)
            with torch.no_grad():
                logits = model(tensor)
        except Exception:
            if device.type != "cuda":
                raise
            logger.exception("Disease model GPU inference failed; retrying on CPU.")
            device = torch.device("cpu")
            model = model.to(device)
            tensor = tensor.to(device)
            with torch.no_grad():
                logits = model(tensor)

        if logits.ndim != 2 or logits.shape != (1, len(PATHOLOGY_LABELS)):
            raise InferenceError("The disease model returned an unexpected output shape.")
        probabilities = torch.sigmoid(logits)[0].detach().cpu().numpy()
        raw_logits = logits[0].detach().cpu().numpy()
        if not np.isfinite(probabilities).all() or not np.isfinite(raw_logits).all():
            raise InferenceError("The disease model returned non-finite scores.")

        predictions = tuple(
            Prediction(label=label, score=float(score), threshold=thresholds[label])
            for label, score in zip(PATHOLOGY_LABELS, probabilities)
        )
        return AnalysisResult(
            model_name=MODEL_NAME,
            checkpoint_id=DISEASE_MODEL_CHECKPOINT.name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            predictions=predictions,
            score_semantics=(
                "Independent sigmoid scores from 14 raw logits; per-disease cutoffs are validation-set "
                "Youden's J thresholds. Scores are not calibrated probabilities or diagnoses."
            ),
            raw_scores=tuple(float(value) for value in raw_logits),
            pathologies=PATHOLOGY_LABELS,
            image_hash=image_hash,
            image_fingerprint=fingerprint_image(image),
            model_revision=model_revision,
        )
    except (ModelUnavailableError, InferenceError):
        raise
    except Exception as exc:
        logger.exception("Project DenseNet-121 disease inference failed.")
        raise InferenceError(
            "Disease analysis could not be completed. Check the image and local model files."
        ) from exc
