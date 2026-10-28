# LungLens AI

LungLens AI is an educational/research Streamlit prototype for exploring chest-X-ray input validation, multi-label model scores, and Grad-CAM attribution. It is **not a medical device or clinical diagnostic system** and must not be used to make medical decisions.

## Pipeline

1. Validate uploaded PNG/JPEG contents, decode the image, apply EXIF orientation, and enforce upload/image size limits.
2. Run the local MobileNetV2 `xray_classifier.pth` suitability gate. Its score must reach `LUNGLENS_XRAY_THRESHOLD` (default `0.8`) before disease inference is allowed. The gate is re-run when analysis is requested; a non-pass or model error blocks inference, Grad-CAM, and PDF export.
3. Run the project-provided 14-label DenseNet-121 on the same image, using the exact inference preprocessing supplied with the model.
4. Apply each disease's own threshold to its independent sigmoid score. Grad-CAM is generated for a selected logit and is an attribution visualization, not a disease boundary.

The image is bound to the validation and analysis state by content hash and normalized-image fingerprint. Upload changes, validation failures, or checkpoint/threshold-file changes clear stale results.

## Model files

Place both checkpoints in the locations shown:

- `xray_classifier.pth` in the project root: MobileNetV2 image suitability gate (`non_xray`, `xray`).
- `Disese_model/my_densenet121.pt`: fine-tuned torchvision DenseNet-121 raw state dict with 14 outputs.
- `Disese_model/thresholds_my.json`: per-label validation thresholds and threshold metadata.

The disease checkpoint is loaded with `torch.load(..., weights_only=True)`, installed into `torchvision.models.densenet121(weights=None)` with a `Linear(1024, 14)` classifier, and loaded with `strict=True`. No pretrained download is needed for this local disease model.

### Disease labels and score interpretation

Output order is fixed to the checkpoint's training order:

`Atelectasis`, `Cardiomegaly`, `Consolidation`, `Edema`, `Effusion`, `Emphysema`, `Fibrosis`, `Hernia`, `Infiltration`, `Mass`, `Nodule`, `Pleural_Thickening`, `Pneumonia`, `Pneumothorax`.

The 14 model outputs are **raw logits**; the application applies an independent sigmoid to each. They are not softmaxed and do not sum to one. Each score is compared with its matching cutoff in `thresholds_my.json`, selected using Youden's J on the validation set. The UI and PDF distinguish labels meeting their cutoffs from those below. There is no `Normal` model class: no cutoff crossings do not establish that an image is normal.

The project-owner-provided test summary in `Disese_model/test_results.csv` reports mean AUROC `0.799` and mean AUPRC `0.253` across the 14 labels. These are owner-reported aggregate evaluation results, not a clinical validation. The owner notes that several cutoffs favor sensitivity and can have low specificity (for example, reported test specificity about `0.39` for Consolidation, `0.50` for Infiltration, and `0.53` for Effusion). Hernia has few positives and its cutoff is particularly uncertain. Scores are not necessarily calibrated probabilities.

The owner reports training with an ImageNet-initialized torchvision DenseNet-121 on NIH ChestX-ray14 using the official train/test split and a patient-level validation split. The checkpoint is the reported validation-best epoch (epoch 6; validation mean AUROC `0.8285`). These training and evaluation details were supplied by the project owner and have not been independently reproduced here. The separate Hugging Face model mentioned during model development is not used by this application. The owner reports NIH ChestX-ray14 as CC0/public domain based on the Kaggle dataset page; verify the dataset's applicable terms independently. Torchvision ImageNet weight terms may also apply; confirm licensing before commercial use. No license statement for the project-trained checkpoint was supplied.

### Exact disease-model preprocessing

For inference, the pipeline is:

1. `image.convert("L")`
2. Resize to `320 × 320` using PIL bilinear interpolation
3. Convert grayscale image to RGB (three repeated channels)
4. `ToTensor()`
5. Normalize using mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`

Training-only augmentations (random resized crop, rotation, and color jitter) and horizontal flips are not used at inference.

## Suitability gate caveat

The X-ray/non-X-ray model is separate from the disease model. The disease model only runs after a matching gate pass. The gate checkpoint's training data, license, and independent performance evaluation were not supplied. Its decision is not a guarantee of authenticity, frontal view, or clinical usability. A false gate rejection can prevent disease analysis.

## Setup

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Set a different X-ray gate threshold before launch if needed:

```powershell
$env:LUNGLENS_XRAY_THRESHOLD = "0.8"
streamlit run app.py
```

This setting affects only the image-suitability gate. Disease cutoffs are read separately from `Disese_model/thresholds_my.json`.

## Tests

Run:

```powershell
python -m pytest -v
```

Tests cover file validation, gate enforcement, strict disease-checkpoint loading, preprocessing, sigmoid/threshold application, Grad-CAM gating, report generation, and Streamlit workflow. Mock-gate tests verify application control flow, not model accuracy.

## Limitations

- The model is for education/research, not clinical use. Scores do not confirm or exclude disease.
- Per-label validation cutoffs can produce false positives and false negatives; the Hernia cutoff is especially uncertain.
- Grad-CAM indicates model attribution at coarse spatial resolution. It does not show exact lesion boundaries or prove why the score was produced.
- Performance may differ across institutions, populations, image devices, or acquisition protocols.
- Training and model artifacts were supplied by the project owner, and their reported performance has not been independently reproduced. The application keeps the current upload in session memory and does not write uploaded image contents to a persistent store; use de-identified images only.
