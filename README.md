# LungLens AI

LungLens AI is a Streamlit research and education prototype for exploring chest-X-ray input validation, multi-label model outputs, and Grad-CAM attribution. It is **not a medical device or clinical diagnostic system**. Do not use its scores, visualizations, or reports to make medical decisions or to replace review by a qualified clinician.

## What the app does

The app has three main areas:

- **Overview** introduces the workflow and project.
- **Analyze X-Ray** accepts one image, checks it, runs the gated disease model, displays scores and class-specific attribution, and offers a PDF research dossier.
- **Model Architecture & Methodology** describes the checkpoints, preprocessing, threshold interpretation, explainability, and limitations.

The analysis always runs in this order:

1. **File validation:** check the extension and actual PNG/JPEG contents, decode the image, apply EXIF orientation, and enforce upload and pixel limits.
2. **Chest-X-ray suitability gate:** use the separate local MobileNetV2 checkpoint. Disease analysis is blocked unless the gate accepts the same image.
3. **Disease inference:** run the project-provided 14-label DenseNet-121 and apply a separate sigmoid to each output logit.
4. **Threshold comparison:** compare each score with that label's configured cutoff. Only labels meeting their own cutoff are treated as threshold-crossing findings.
5. **Attribution and report:** generate per-label Grad-CAM maps for threshold-crossing findings when a usable map is available. The PDF pairs the uploaded X-ray with each finding's map, score/cutoff, and general educational description. If a map cannot be generated, the PDF identifies it as unavailable instead of inventing an image.

The gate is checked again when the user requests disease analysis. A rejection, inability to verify the image, missing validation service, stale image, or mismatched validation result blocks downstream inference, Grad-CAM, and PDF export.

## User workflow

1. Open **Analyze X-Ray** and upload one PNG, JPG, or JPEG image. The upload panel can also load the bundled `assets/Xray sample.jpg`; it goes through the same gate as an uploaded image.
2. The uploader is replaced by **Upload another file** while an image is loaded. Selecting it clears the current image and its associated validation, analysis, heatmaps, and report state, then brings the uploader back.
3. Review the file-validation and chest-X-ray suitability status. Passing the gate permits research analysis but does not guarantee that the image is authentic, frontal, or clinically usable.
4. Select **Run DenseNet-121 Analysis**. The app rechecks suitability before calling the disease model.
5. Review the 14 independent scores and their per-label cutoffs. The Grad-CAM selector contains only cutoff-crossing labels and orders them from highest score to lowest.
6. When analysis is available, **Download Research Dossier (PDF)** appears next to the analysis button. The report includes all score/cutoff rows and a paired original/attribution panel for each threshold-crossing finding. Missing Grad-CAM maps are explicitly labelled; no map is fabricated.

If no disease score crosses its cutoff, the app says so. This is **not** a `Normal` prediction: the disease model has no normal class, and no cutoff crossing does not rule out disease.

## Requirements

- Windows, macOS, or Linux
- Python 3.10 or later; Python 3.11 is a suitable choice for a new environment
- The project disease-model assets listed below
- The owner-created `xray_classifier.pth` suitability checkpoint in the project root

Dependencies are listed in [`requirements.txt`](./requirements.txt). PyTorch and Torchvision must have compatible builds. The requirements file sets minimum versions, but if you need CUDA, install a PyTorch/Torchvision pair appropriate for your CUDA runtime using the official PyTorch installation instructions before installing or reconciling the remaining requirements.

## Model files

### Disease model (included in this repository)

These project-owned assets are expected at these exact paths:

| File | Purpose |
|---|---|
| `Disease_model/my_densenet121.pt` | Raw PyTorch state dictionary for the fine-tuned 14-output DenseNet-121 |
| `Disease_model/thresholds_my.json` | Per-label cutoffs and supplied threshold metadata |
| `Disease_model/explanations.py` | General educational descriptions shown alongside threshold-crossing findings |
| `Disease_model/test_results.csv` | Project-owner-provided aggregate evaluation summary |
| `Disease_model/curves.png` | Supplied evaluation-curve image |

The disease model is created as `torchvision.models.densenet121(weights=None)`, with a `Linear(1024, 14)` classifier. The raw state dictionary is loaded with `torch.load(..., weights_only=True)` and `strict=True`; no ImageNet checkpoint download is required at runtime.

### X-ray suitability gate (owner-created checkpoint)

The gate checkpoint is expected at:

```text
xray_classifier.pth
```

This is the project owner's trained X-ray-versus-normal-photo classifier checkpoint and is included in this repository. The `.gitignore` does not exclude it. The application loads the checkpoint into a MobileNetV2 model with a two-output classifier. It expects a PyTorch checkpoint dictionary containing:

- `state_dict`: MobileNetV2 weights matching the app's two-output classifier
- `classes`: a two-item sequence containing `non_xray` and `xray`

The app uses the class names in the checkpoint to find the X-ray score; do not change the checkpoint's class metadata or weights independently. If this file is missing, invalid, or incompatible with the installed PyTorch/Torchvision stack, the gate reports that its service is unavailable and blocks disease analysis.

The gate is separate from the DenseNet disease model. It does not use a classical image-feature rule or the disease scores to decide whether to run pathology inference.

## Image validation and suitability

### Stage 1: file validation

Supported file extensions and contents are PNG, JPG, and JPEG. The file is decoded using Pillow; a misleading extension does not make unsupported image contents acceptable. EXIF orientation is applied before the image is normalized to RGB. The current limits in `config.py` are:

- Maximum upload size: **10 MiB**
- Maximum decoded image area: **20,000,000 pixels**

Empty, oversized, excessive-dimension, unsupported, or corrupt files are rejected with a file-validation message. For 16-bit/integer/float images that Pillow decodes, values are rescaled to an 8-bit image before RGB conversion.

### Stage 2: chest-X-ray suitability

The local MobileNetV2 gate converts the decoded image to RGB, resizes it to `224 × 224`, converts it to a tensor, and normalizes it with ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`. The app takes the softmax score corresponding to the checkpoint's `xray` class and compares it with `LUNGLENS_XRAY_THRESHOLD`, which defaults to `0.8`.

The gate's training set, independent performance evaluation, calibration, and license were not provided. Its threshold is configurable, not clinically validated, and not a reliable guarantee that every photograph, screenshot, manipulated image, or non-frontal view will be detected. A pass is not proof of authenticity or clinical suitability; a false rejection can also prevent analysis.

The UI distinguishes acceptance, rejection, inability to verify, and unavailable service. In every non-accepted state, pathology scores are not displayed and inference is blocked.

## Disease model outputs

The model has 14 independent outputs in this fixed order:

1. Atelectasis
2. Cardiomegaly
3. Consolidation
4. Edema
5. Effusion
6. Emphysema
7. Fibrosis
8. Hernia
9. Infiltration
10. Mass
11. Nodule
12. Pleural_Thickening
13. Pneumonia
14. Pneumothorax

The checkpoint returns **raw logits**. The application applies `sigmoid` independently to each logit, not softmax. Scores do not sum to one and must not be interpreted as calibrated clinical probabilities. A score meeting its configured cutoff is a threshold crossing for research display, not a diagnosis.

The supplied validation-set Youden's J cutoffs are:

| Label | Cutoff |
|---|---:|
| Atelectasis | 0.225708 |
| Cardiomegaly | 0.049957 |
| Consolidation | 0.121277 |
| Edema | 0.047607 |
| Effusion | 0.257568 |
| Emphysema | 0.053986 |
| Fibrosis | 0.027328 |
| Hernia | 0.010000 |
| Infiltration | 0.438721 |
| Mass | 0.134521 |
| Nodule | 0.172485 |
| Pleural_Thickening | 0.074646 |
| Pneumonia | 0.055817 |
| Pneumothorax | 0.114990 |

The application reads the values from `Disease_model/thresholds_my.json`; the JSON file is the runtime source of truth. They are shown here to make the default model configuration transparent. The project owner reports that these cutoffs were selected on validation data, not the test set.

### Exact disease-model inference preprocessing

1. Convert the oriented image to grayscale: `image.convert("L")`.
2. Resize to `320 × 320` using PIL bilinear resampling.
3. Convert the grayscale image to RGB (three repeated channels).
4. Convert to a `[0, 1]` tensor.
5. Normalize using ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.

Training-only random resized crop, rotation, color jitter, and any horizontal flip are not applied at inference.

## Model provenance and reported evaluation

According to details supplied by the project owner, the disease checkpoint is a torchvision DenseNet-121 initialized from ImageNet weights and fine-tuned on NIH ChestX-ray14. The owner reports using the official NIH train/test split, a patient-level validation split, and selecting epoch 6 as the best checkpoint by validation mean AUROC (`0.8285`). These training details have not been independently reproduced in this repository. The separate X-ray suitability checkpoint was trained by the project owner and is included as `xray_classifier.pth`; its training dataset, evaluation metrics, and formal license were not supplied, so its detection performance is not independently established here.

The owner-provided `Disease_model/test_results.csv` reports mean test AUROC `0.799` and mean AUPRC `0.253` across the 14 labels. These are owner-reported aggregate figures, not a clinical validation or a guarantee of performance on another population or institution. The owner reports low specificity for some labels (approximately `0.39` for Consolidation, `0.50` for Infiltration, and `0.53` for Effusion) at the supplied cutoffs. The Hernia cutoff is especially uncertain because of the small number of positive examples reported for validation.

The project owner reports NIH ChestX-ray14 as CC0/public domain based on the Kaggle dataset page; verify the dataset's applicable terms independently. Torchvision/ImageNet terms may also apply. The project owner trained the gate checkpoint; a formal license statement for the trained checkpoints was not supplied. Verify applicable rights before redistribution or commercial use.

## Grad-CAM and PDF report

Grad-CAM uses the selected disease logit and DenseNet feature maps. The selector only offers diseases whose sigmoid score meets the supplied per-label cutoff, sorted in descending score order. Selecting a different disease displays that label's corresponding attribution map and general educational reference, when available.

Grad-CAM is a coarse model-attribution visualization, **not** an exact lesion segmentation or proof of pathology. A finding can meet its score cutoff yet produce no usable positive attribution map. In that case the UI and PDF report that the attribution is unavailable; they do not substitute another disease's map.

The PDF report is generated only for an accepted suitability result and the exact image associated with the analysis. It contains model/checkpoint metadata, gate metadata, all 14 scores with thresholds and cutoff statuses, a list of threshold-crossing labels, and one side-by-side provided-image/Grad-CAM section per crossing, including the score, cutoff, and general educational description. If a map is unavailable, the report includes an explicit note in its place. If there are no threshold crossings, no disease-specific attribution panel is presented.

## Installation and launch

### Windows PowerShell

From the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The owner-created `xray_classifier.pth` is included in the repository root. Start the app:

```powershell
streamlit run app.py
```

Streamlit normally serves the app at `http://localhost:8501`. If that port is already in use:

```powershell
streamlit run app.py --server.port 8502
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# The owner-created xray_classifier.pth checkpoint is included in the repository root.
streamlit run app.py
```

### Configure the gate threshold

The default suitability threshold is `0.8`. To override it for a local research experiment, set the environment variable before launching the app.

Windows PowerShell:

```powershell
$env:LUNGLENS_XRAY_THRESHOLD = "0.8"
streamlit run app.py
```

macOS or Linux:

```bash
LUNGLENS_XRAY_THRESHOLD=0.8 streamlit run app.py
```

The value must be between `0` and `1`. Changing it affects only the input-suitability gate. Disease cutoffs remain in `Disease_model/thresholds_my.json`.

## Running tests

Run the full suite from the repository root:

```powershell
python -m pytest -v
```

Or, on macOS/Linux:

```bash
python -m pytest -v
```

The latest recorded full test run completed with **54 passed** before the directory and bundled sample-image updates in this revision. Tests cover image decoding and file limits, suitability states and gating, checkpoint loading, preprocessing, sigmoid and cutoff behavior, state clearing after upload changes, Grad-CAM gating, PDF content/export, and Streamlit workflow. App logic tests use mocked gate decisions and synthetic inputs where appropriate; those tests verify software control flow, **not classifier accuracy**. Checkpoint smoke tests verify that model weights load and return outputs of the expected shape; they do not prove classifier accuracy.

## Troubleshooting

### “Image validation service unavailable”

- Confirm `xray_classifier.pth` exists in the project root.
- Confirm it contains both `state_dict` and `classes`, with class names `non_xray` and `xray`.
- Confirm PyTorch and Torchvision are installed with compatible versions.
- The app intentionally blocks disease inference if the gate is unavailable.

### “Unable to verify image type” or image rejected

- Try a frontal chest X-ray in PNG or JPEG format.
- Rejection means the gate score was below the configured cutoff; it is not a clinical opinion.
- The owner-created gate has not been independently evaluated here. Its pass/reject result is not a guarantee of modality.
- The bundled `assets/Xray sample.jpg` is the sample image loaded by the UI and is evaluated by the same gate as uploaded files. A rejection is not bypassed just because the image is bundled.

### Disease model unavailable

- Confirm `Disease_model/my_densenet121.pt` and `Disease_model/thresholds_my.json` exist and are readable.
- Confirm the checkpoint matches the documented DenseNet-121 architecture and 14-label output order.
- Ensure the Python environment can import compatible PyTorch and Torchvision builds.

### Grad-CAM unavailable for a finding

The selected output may have no usable positive gradient attribution map for that image. The finding score is not changed by this; the app marks the attribution unavailable and will not fabricate a heatmap.

### PDF generation error

Make sure the analysis and validation belong to the currently loaded image, then retry. The app displays and logs the specific report-generation error so it can be diagnosed; PDF export remains behind the accepted-image gate.

## Privacy and limitations

- Uploaded image bytes, analysis outputs, and report data are held in the active Streamlit session; the app code does not deliberately save the uploaded image to a persistent application store. Hosting and deployment infrastructure may have separate retention or logging behavior.
- De-identify images before uploading. Do not upload protected health information to an environment that is not approved to handle it.
- Model outputs can be false positives or false negatives and are not calibrated diagnostic probabilities.
- A gate pass is not proof that an image is a genuine, unaltered, frontal, or clinically interpretable radiograph.
- The reported disease-model evaluation is project-owner supplied and may not generalize to different scanners, institutions, populations, or protocols.
- Grad-CAM highlights model attribution at limited spatial resolution; it is not a lesion boundary.
- Use only for education and research. Seek qualified clinical interpretation for medical concerns.
