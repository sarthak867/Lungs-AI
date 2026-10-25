# LungLens AI

LungLens AI is an educational and research-oriented prototype for exploring pretrained chest X-ray model outputs and conditional visual attribution. It is **not a clinical diagnostic system** and must not be used to make medical decisions.

## Features

- In-memory PNG/JPG/JPEG upload validation and aspect-ratio-preserving preview
- Separate file-integrity validation and local `xray_classifier.pth` chest-X-ray suitability gate
- TorchXRayVision pretrained DenseNet-121 pathology inference, only after the matching local X-ray gate passes
- Model-label-driven output display with explicit score semantics
- Grad-CAM only when valid gradients and a supported target layer are available
- Downloadable PDF reports for successfully gated analyses
- Research-use disclaimers throughout the interface

## Architecture

- `app.py`: Streamlit entry point and page flow
- `components/`: sidebar and prediction display
- `services/preprocessing.py`: file validation and DenseNet input normalization
- `services/modality_validation.py`: suitability decisions and image-fingerprint binding
- `services/xray_gate_model.py`: safe local checkpoint loading and image classification; the only modality classifier used by the app
- `services/upload_state.py`: upload identity and stale-result invalidation
- `services/inference.py`: TorchXRayVision model loading and pathology inference
- `services/explainability.py`: conditional Grad-CAM generation
- `services/report_generator.py`: ReportLab PDF output
- `tests/`: file, gate, inference, report, and Streamlit workflow tests

## Local X-ray suitability classifier

The gate uses only the project-owner-provided `xray_classifier.pth` with the torchvision MobileNetV2 architecture. It reads weights from `checkpoint["state_dict"]`, reads the class names from `checkpoint["classes"]`, replaces the classifier head with `Dropout(0.3)` followed by a two-output linear layer, strictly loads the state dict, and sets the model to evaluation mode. The expected classes are `["non_xray", "xray"]`; the X-Ray score is selected by `classes.index("xray")`. No classical computer-vision heuristics, handcrafted image features, or other modality classifier are combined with or used to override its output. The checkpoint is loaded locally and is not downloaded.

Preprocessing follows the transform supplied for training: RGB conversion, resize to 224 × 224, tensor conversion, and normalization with mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`. Grayscale uploads are converted to three identical color channels.

Preprocessing follows the training transform: retain source color, convert to RGB, resize to 224 × 224, tensor conversion, and ImageNet normalization with mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`. 16-bit/integer/float grayscale inputs are normalized to 8-bit before RGB conversion. The X-Ray score is a two-class softmax score and is not assumed calibrated. The configurable `LUNGLENS_XRAY_THRESHOLD` defaults to `0.8`; scores below it are rejected and block pathology inference. This is not a threshold validated for accuracy. The app re-runs this classifier when the analysis action is requested instead of trusting only the earlier upload-time decision. Cached upload decisions are invalidated when checkpoint size, modification time, or threshold changes.

The checkpoint was supplied by the project owner. Its training dataset, license, and independent evaluation metrics were not supplied. In the observed local smoke check, strict MobileNetV2 loading succeeded, but the bundled `assets/sample_xray.png` received an X-Ray score of `0.0183` and was rejected at the `0.8` threshold; a generated RGB photo-like fixture received `0.0012` and was also rejected. Thus, the required X-Ray acceptance check did **not** pass on the bundled sample, and this checkpoint must not be treated as demonstrated to distinguish X-rays reliably. Mock pass/reject tests exercise application logic only; synthetic images are **not evidence of classifier accuracy**. The model may misclassify photographs, screenshots, drawings, unusual radiographs, or manipulated images. A pass cannot guarantee image authenticity, frontal view, or clinical usability.

## Pathology model and preprocessing

After the local gate passes, the app loads TorchXRayVision's `xrv.models.DenseNet(weights="densenet121-res224-all")`. The pathology input follows that project's documented example: grayscale conversion, centered square crop, resize to 224 × 224, and normalization to approximately [-1024, 1024]. Labels come from `model.pathologies`. The checkpoint is pretrained and was not trained by this project.

## Screenshots

No screenshot files are committed in this repository yet.

## Prerequisites

- Python 3.11 is recommended.
- CPU is sufficient; GPU is optional.
- Place `xray_classifier.pth` in the project root. It is intentionally excluded from Git.
- Initial pathology inference may download the official TorchXRayVision checkpoint and requires network access.

## Windows PowerShell setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Start the application:

```powershell
streamlit run app.py
```

The local MobileNetV2 checkpoint classifier runs first. TorchXRayVision pathology weights load only after the X-Ray class score reaches the configured threshold. Model download or initialization failures are reported instead of producing substitute predictions.

Optionally configure the two-class decision boundary before launch:

```powershell
$env:LUNGLENS_XRAY_THRESHOLD = "0.8"
streamlit run app.py
```

The default is the two-class argmax boundary, not a validated operating threshold.

## Tests

```powershell
python -m pytest -v
```

Mock suitability decisions test application control flow, not classifier accuracy. The local checkpoint smoke test verifies that provided weights load and return a finite, bounded X-Ray score. Neither is a modality-detection accuracy evaluation.

## Troubleshooting

- **Image validation service unavailable:** ensure `xray_classifier.pth` is in the project root and contains the expected MobileNetV2 `state_dict` and `classes` entries.
- **Model unavailable:** verify Torch/Torchvision installation and network access for first-time TorchXRayVision checkpoint download.
- **Upload rejected:** use a readable PNG/JPG/JPEG no larger than 10 MB or 20 megapixels.
- **Unable to verify:** the classifier failed or scored exactly at the configured boundary; analysis remains blocked.
- **No Grad-CAM:** the architecture, target layer, or gradients may not support a valid attribution map; the app does not synthesize one.
- **PowerShell activation blocked:** use `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` and `.\.venv\Scripts\streamlit.exe run app.py` without activating the environment.

## Limitations and intended use

The local classifier's provenance beyond the supplied checkpoint, training dataset, license, and measured performance were not provided. The softmax score is not necessarily a calibrated probability. The pathology model also has no clinical target-population evaluation from this project. Grad-CAM is an attribution visualization, not proof that a highlighted region contains disease. Uploaded image contents are kept in Streamlit session memory and are not saved by the app; avoid uploading identifiable patient information.

## References and acknowledgements

- [TorchXRayVision documentation and source](https://github.com/mlmed/torchxrayvision)
- [TorchXRayVision model documentation](https://mlmed.org/torchxrayvision/)
- [TorchXRayVision upstream inference example](https://github.com/mlmed/torchxrayvision#models)
- [Streamlit documentation](https://docs.streamlit.io/)
- Project-owner-provided `xray_classifier.pth` (training and license details were not supplied)
