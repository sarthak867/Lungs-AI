from __future__ import annotations

from io import BytesIO

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import HRFlowable, Image as ReportImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from Disese_model.explanations import DISCLAIMER, EXPLANATIONS
from services.inference import AnalysisResult
from services.modality_validation import (
    SuitabilityValidationResult,
    SuitabilityGateError,
    require_suitability_pass,
)


def generate_pdf_report(
    result: AnalysisResult,
    filename: str,
    *,
    original_image: Image.Image | None = None,
    gradcam_images: dict[str, Image.Image] | None = None,
    image_hash: str,
    validation: SuitabilityValidationResult | None,
) -> bytes:
    if not isinstance(original_image, Image.Image):
        raise SuitabilityGateError("A report requires the original image associated with the validated analysis.")
    if gradcam_images is not None and (
        not isinstance(gradcam_images, dict)
        or any(not isinstance(image, Image.Image) for image in gradcam_images.values())
    ):
        raise TypeError("Grad-CAM images must be a mapping of pathology labels to PIL images.")
    require_suitability_pass(validation, image_hash, original_image)
    if result.image_hash != image_hash or result.image_fingerprint != validation.image_fingerprint:
        raise SuitabilityGateError("A report can only be generated for the validated analysis image.")
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
        title="LungLens AI Research Report",
    )
    styles = getSampleStyleSheet()

    # Custom styling
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0A192F"),
        alignment=0,
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#0284C7"),
        spaceAfter=8,
    )
    meta_style = ParagraphStyle(
        "DocMeta",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#334155"),
    )
    disclaimer_style = ParagraphStyle(
        "DocDisclaimer",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#64748B"),
    )
    h2_style = ParagraphStyle(
        "DocH2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=10,
        spaceAfter=6,
    )

    story = [
        Paragraph("LungLens AI — Radiograph Research Dossier", title_style),
        Paragraph("PROJECT CHEST X-RAY MULTI-LABEL INFERENCE & ATTRIBUTION", subtitle_style),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0284C7"), spaceAfter=10),
        Paragraph(
            "<b>Research prototype evaluation only:</b> This dossier is generated for educational and algorithm research. "
            "It does not constitute medical advice or a verified clinical diagnosis.",
            disclaimer_style,
        ),
        Spacer(1, 8),
    ]

    # Metadata summary grid
    meta_data = [
        [
            Paragraph(f"<b>Timestamp:</b> {result.timestamp}", meta_style),
            Paragraph(f"<b>Source Image:</b> {_escape(filename)}", meta_style),
        ],
        [
            Paragraph(f"<b>Model:</b> {result.model_name}", meta_style),
            Paragraph(f"<b>Checkpoint:</b> {result.checkpoint_id}", meta_style),
        ],
        [
            Paragraph(f"<b>Score Semantics:</b> {_escape(result.score_semantics)}", meta_style),
            Paragraph("<b>Explainability:</b> Grad-CAM (denseblock4)", meta_style),
        ],
        [
            Paragraph(
                f"<b>Input gate:</b> {_escape(validation.classifier_id or 'Local X-ray classifier')}",
                meta_style,
            ),
            Paragraph(
                "<b>X-Ray class score:</b> "
                + (
                    f"{validation.xray_score:.4f} (not calibrated)"
                    if validation.xray_score is not None
                    else "Unavailable"
                ),
                meta_style,
            ),
        ],
    ]
    meta_table = Table(meta_data, colWidths=[3.7 * inch, 3.5 * inch])
    meta_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#F1F5F9")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.extend([meta_table, Spacer(1, 10), Paragraph("Independent Sigmoid Outputs", h2_style)])
    if result.findings:
        finding_text = ", ".join(_escape(item.label.replace("_", " ")) for item in result.findings)
        story.append(Paragraph(f"<b>Outputs meeting their supplied cutoffs:</b> {finding_text}", meta_style))
    else:
        story.append(
            Paragraph(
                "<b>No output crossed its supplied per-disease cutoff.</b> This does not establish that the image is normal.",
                meta_style,
            )
        )
    story.append(Spacer(1, 6))

    rows = [["Pathology label", "Sigmoid score", "Val cutoff", "Cutoff status"]]
    for item in result.predictions:
        cutoff_status = "Meets cutoff" if item.meets_threshold else "Below cutoff"
        threshold = f"{item.threshold:.4f}" if item.threshold is not None else "Not set"
        rows.append(
            [
                _escape(item.label.replace("_", " ")),
                f"{item.score:.5f}",
                threshold,
                cutoff_status,
            ]
        )

    if len(rows) == 1:
        rows.append(["No supported outputs were returned", "", ""])

    table = Table(rows, colWidths=[2.5 * inch, 1.45 * inch, 1.25 * inch, 1.95 * inch], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B2238")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#38BDF8")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 8.5),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#0F172A")),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 1), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.extend([table, Spacer(1, 12)])

    gradcam_images = gradcam_images or {}

    if result.findings:
        story.append(Paragraph("Threshold-crossing findings and per-label Grad-CAM", h2_style))
        for finding in result.findings:
            label = finding.label.replace("_", " ")
            description = EXPLANATIONS.get(finding.label, "No educational imaging reference is available.")
            gradcam_image = gradcam_images.get(finding.label)
            attribution_panel = (
                _reportlab_image(gradcam_image, max_width=3.35 * inch)
                if gradcam_image is not None
                else Paragraph(
                    "<b>Grad-CAM unavailable.</b> No usable positive attribution map was produced for this finding.",
                    disclaimer_style,
                )
            )
            image_pair = Table(
                [
                    [
                        Paragraph("<b>Provided X-ray</b>", meta_style),
                        Paragraph(
                            f"<b>Grad-CAM · {_escape(label)}</b><br/>"
                            f"Score: {finding.score:.4f} · Cutoff: {finding.threshold:.4f}",
                            meta_style,
                        ),
                    ],
                    [
                        _reportlab_image(original_image, max_width=3.35 * inch),
                        attribution_panel,
                    ],
                    [
                        Paragraph(
                            f"<b>General educational reference:</b> {_escape(description)} "
                            f"{_escape(DISCLAIMER)}",
                            disclaimer_style,
                        ),
                        "",
                    ],
                ],
                colWidths=[3.55 * inch, 3.55 * inch],
            )
            image_pair.setStyle(
                TableStyle(
                    [
                        ("SPAN", (0, 2), (1, 2)),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("GRID", (0, 0), (-1, 1), 0.4, colors.HexColor("#CBD5E1")),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
                        ("LEFTPADDING", (0, 0), (-1, -1), 6),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                        ("TOPPADDING", (0, 0), (-1, -1), 6),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ]
                )
            )
            story.extend([image_pair, Spacer(1, 10)])
    else:
        story.append(
            Paragraph(
                "No output crossed its supplied per-disease cutoff; no disease-specific Grad-CAM is shown.",
                meta_style,
            )
        )

    story.append(
        Paragraph(
            "<b>Attribution Notice:</b> Highlighted areas reflect convolutional receptive field activation gradients and "
            "do not prove pathological anatomical boundaries. Performance on arbitrary populations has not been clinically validated.",
            disclaimer_style,
        )
    )

    document.build(story)
    return buffer.getvalue()


def _reportlab_image(image: Image.Image, *, max_width: float = 4.8 * inch) -> ReportImage:
    image_buffer = BytesIO()
    image.convert("RGB").save(image_buffer, format="PNG")
    image_buffer.seek(0)
    width = max_width
    height = width * image.height / image.width
    maximum_height = 3.2 * inch
    if height > maximum_height:
        width *= maximum_height / height
        height = maximum_height
    return ReportImage(image_buffer, width=width, height=height)


def _escape(value: str) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
