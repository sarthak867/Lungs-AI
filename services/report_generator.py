from __future__ import annotations

from io import BytesIO

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import HRFlowable, Image as ReportImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from services.inference import AnalysisResult
from services.modality_validation import (
    SuitabilityValidationResult,
    SuitabilityGateError,
    require_suitability_pass,
)


def generate_pdf_report(
    result: AnalysisResult,
    filename: str,
    original_image: Image.Image | None = None,
    gradcam_image: Image.Image | None = None,
    *,
    image_hash: str,
    validation: SuitabilityValidationResult | None,
) -> bytes:
    if original_image is None:
        raise SuitabilityGateError("A report requires the original image associated with the validated analysis.")
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
        Paragraph("PRETRAINED CHEST X-RAY MULTI-LABEL INFERENCE & ATTRIBUTION", subtitle_style),
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
    story.extend([meta_table, Spacer(1, 10), Paragraph("Model Pathology Outputs", h2_style)])

    rows = [["Pathology Finding", "Activation Score", "Relative Index"]]
    max_score = max((abs(p.score) for p in result.predictions), default=1.0) or 1.0
    for idx, item in enumerate(result.predictions, start=1):
        rel_pct = (abs(item.score) / max_score) * 100
        rows.append([_escape(item.label), f"{item.score:.5f}", f"{rel_pct:.1f}% relative"])

    if len(rows) == 1:
        rows.append(["No supported outputs were returned", "", ""])

    table = Table(rows, colWidths=[3.4 * inch, 1.9 * inch, 1.9 * inch], repeatRows=1)
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

    # Images section
    for title, image in (
        ("Input Radiograph (224 × 224 center-crop geometry preview)", original_image),
        ("Grad-CAM Attribution Overlay (denseblock4 gradient localization)", gradcam_image),
    ):
        if image is not None:
            story.append(Paragraph(title, h2_style))
            story.append(_reportlab_image(image))
            story.append(Spacer(1, 8))

    story.append(
        Paragraph(
            "<b>Attribution Notice:</b> Highlighted areas reflect convolutional receptive field activation gradients and "
            "do not prove pathological anatomical boundaries. Performance on arbitrary populations has not been clinically validated.",
            disclaimer_style,
        )
    )

    document.build(story)
    return buffer.getvalue()


def _reportlab_image(image: Image.Image) -> ReportImage:
    image_buffer = BytesIO()
    image.convert("RGB").save(image_buffer, format="PNG")
    image_buffer.seek(0)
    width = 4.8 * inch
    height = width * image.height / image.width
    maximum_height = 3.2 * inch
    if height > maximum_height:
        width *= maximum_height / height
        height = maximum_height
    return ReportImage(image_buffer, width=width, height=height)


def _escape(value: str) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
