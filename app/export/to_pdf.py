from __future__ import annotations

from pathlib import Path
from typing import Dict

import cv2
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

from app.models import ProcessingResult


def export_pdf(result: ProcessingResult, export_dir: Path, meta: Dict[str, str] | None = None) -> Path:
    """Create a PDF report with overlay image and metrics."""
    export_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = export_dir / f"{result.target_id}.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    width, height = A4
    meta = meta or {}
    organization = meta.get("organization", "")
    author = meta.get("author", "")

    c.setTitle(f"Отчёт {result.target_id}")
    c.setAuthor(author)
    top = height - 20 * mm
    c.setFont("Helvetica-Bold", 16)
    c.drawString(20 * mm, top, "Отчёт по мишени")
    c.setFont("Helvetica", 10)
    c.drawString(20 * mm, top - 10 * mm, f"ID: {result.target_id}")
    c.drawString(20 * mm, top - 16 * mm, f"Дата: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    if organization:
        c.drawString(20 * mm, top - 22 * mm, f"Организация: {organization}")

    metrics = result.metrics
    table_top = top - 40 * mm
    if metrics:
        summary = result.to_summary_dict()
        rows = [
            ("Количество", f"{summary.get('point_count', 0)}"),
            ("СТП X (мм)", f"{summary.get('mean_x_mm', 0.0):.2f}"),
            ("СТП Y (мм)", f"{summary.get('mean_y_mm', 0.0):.2f}"),
            ("Смещ. (мм)", f"{summary.get('displacement_mm', 0.0):.2f}"),
            ("Азимут (°)", f"{summary.get('azimuth_deg', 0.0):.2f}"),
            ("Mean Radius", f"{summary.get('mean_radius_mm', 0.0):.2f}"),
            ("Extreme Spread", f"{summary.get('extreme_spread_mm', 0.0):.2f}"),
            ("R50", f"{summary.get('r50_mm', 0.0):.2f}"),
        ]
        c.setFont("Helvetica", 10)
        row_height = 6 * mm
        col1_x = 20 * mm
        col2_x = 70 * mm
        y = table_top
        for name, value in rows:
            c.setFillColor(colors.darkgray)
            c.drawString(col1_x, y, name)
            c.setFillColor(colors.black)
            c.drawString(col2_x, y, value)
            y -= row_height
    image_y = table_top - 70 * mm
    overlay_path = _ensure_overlay_path(result, export_dir)
    if overlay_path and Path(overlay_path).exists():
        c.drawImage(str(overlay_path), 20 * mm, max(40 * mm, image_y), width=120 * mm, height=120 * mm, preserveAspectRatio=True)

    c.showPage()
    c.save()
    return pdf_path


def _ensure_overlay_path(result: ProcessingResult, export_dir: Path) -> str | None:
    if result.overlay_path and Path(result.overlay_path).exists():
        return result.overlay_path
    if result.overlay_image is None:
        return None
    fallback = export_dir / f"{result.target_id}_overlay.png"
    cv2.imwrite(str(fallback), result.overlay_image)
    result.overlay_path = str(fallback)
    return result.overlay_path


__all__ = ["export_pdf"]
