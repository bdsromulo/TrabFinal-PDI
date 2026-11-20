"""Generates the printable A4 PDF Answer Sheet (ReportLab), canonical PNG template, and default answer keys.

Usage:
    python tools/generate_sheet.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

from omr.grade import create_default_exam_config
from omr.sheet import SheetLayout, get_default_layout, render_sheet_image


def generate_pdf_sheet(pdf_path: str | Path, layout: SheetLayout | None = None) -> Path:
    """Renders the printable A4 PDF answer sheet using exact millimeter coordinates from `SheetLayout`."""
    if layout is None:
        layout = get_default_layout()

    target = Path(pdf_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    page_w, page_h = A4  # 210mm x 297mm in points
    c = canvas.Canvas(str(target), pagesize=A4)
    c.setTitle("UTFPR — IF69D / OPEET014 — Folha de Respostas OMR")
    c.setAuthor("Rômulo Barbosa da Silva e Guilherme Oliver")

    def y_pt(y_mm: float) -> float:
        """Converts top-left origin y_mm to ReportLab bottom-left origin points."""
        return (layout.page_height_mm - y_mm) * mm

    # 1. Draw the 4 solid black corner fiducial markers
    c.setFillColorRGB(0, 0, 0)
    c.setStrokeColorRGB(0, 0, 0)
    for m in layout.markers:
        half = m.size_mm / 2.0
        x_left = (m.cx_mm - half) * mm
        y_bottom = y_pt(m.cy_mm + half)
        c.rect(x_left, y_bottom, m.size_mm * mm, m.size_mm * mm, stroke=0, fill=1)

    # 2. Header title
    c.setFillColorRGB(0.1, 0.1, 0.1)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(33 * mm, y_pt(20.5), "UTFPR — IF69D / OPEET014 — FOLHA DE RESPOSTAS (OMR)")

    # Student identification box
    c.setStrokeColorRGB(0.45, 0.45, 0.45)
    c.setLineWidth(0.8)
    c.rect(32 * mm, y_pt(52.0), 92 * mm, 25 * mm, stroke=1, fill=0)

    c.setFont("Helvetica", 9.5)
    c.drawString(35 * mm, y_pt(36.0), "ALUNO(A): _______________________________________________")
    c.drawString(35 * mm, y_pt(47.0), "RA: ________________________    DATA: _____/_____/2026")

    # Exam type box
    c.rect(128 * mm, y_pt(52.0), 52 * mm, 25 * mm, stroke=1, fill=0)
    c.setFont("Helvetica-Bold", 9.5)
    c.drawCentredString(154 * mm, y_pt(33.0), "TIPO DE PROVA")

    c.setLineWidth(1.1)
    c.setStrokeColorRGB(0.2, 0.2, 0.2)
    for b in layout.exam_type_bubbles:
        c.setFont("Helvetica-Bold", 9)
        c.drawCentredString(b.cx_mm * mm, y_pt(b.cy_mm - b.radius_mm - 1.2), b.option)
        c.circle(b.cx_mm * mm, y_pt(b.cy_mm), b.radius_mm * mm, stroke=1, fill=0)

    # 3. Column headers for the 2 columns of 25 questions
    col_x_starts_mm = (46.0, 132.0)
    opt_dx_mm = 9.0
    header_y_mm = 63.0
    c.setFont("Helvetica-Bold", 9.5)
    for col_x0_mm in col_x_starts_mm:
        c.drawCentredString((col_x0_mm - 11.5) * mm, y_pt(header_y_mm), "Q.")
        for opt_idx, opt in enumerate(layout.options):
            cx_mm = col_x0_mm + opt_idx * opt_dx_mm
            c.drawCentredString(cx_mm * mm, y_pt(header_y_mm), opt)

    # 4. Question rows (01..50) and bubbles
    c.setLineWidth(1.0)
    for q_num in range(1, layout.num_questions + 1):
        q_bubbles = layout.get_question_bubbles(q_num)
        first_b = q_bubbles[0]

        c.setFillColorRGB(0.15, 0.15, 0.15)
        c.setFont("Helvetica-Bold", 9.0)
        c.drawCentredString((first_b.cx_mm - 11.5) * mm, y_pt(first_b.cy_mm + 1.1), f"{q_num:02d}")

        for b in q_bubbles:
            c.setStrokeColorRGB(0.22, 0.22, 0.22)
            c.circle(b.cx_mm * mm, y_pt(b.cy_mm), b.radius_mm * mm, stroke=1, fill=0)

    # 5. Footer instructions
    c.setFont("Helvetica", 7.5)
    c.setFillColorRGB(0.35, 0.35, 0.35)
    c.drawCentredString(
        105 * mm,
        y_pt(276.0),
        "Instruções: Preencha totalmente a bolha com caneta esferográfica azul ou preta. "
        "Não rasure e mantenha os 4 quadrados de canto visíveis na foto.",
    )

    c.showPage()
    c.save()
    return target


def main() -> int:
    layout = get_default_layout()
    templates_dir = ROOT_DIR / "data" / "templates"
    keys_dir = ROOT_DIR / "data" / "answer_keys"
    templates_dir.mkdir(parents=True, exist_ok=True)
    keys_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate printable A4 PDF
    pdf_path = generate_pdf_sheet(templates_dir / "answer_sheet.pdf", layout)

    # 2. Save canonical layout JSON
    layout_json_path = templates_dir / "sheet_layout.json"
    layout.save_json(layout_json_path)

    # 3. Render canonical blank PNG reference sheet
    blank_img = render_sheet_image(layout)
    ref_png_path = templates_dir / "reference_sheet.png"
    ok, buf = cv2.imencode(".png", blank_img)
    if ok:
        buf.tofile(str(ref_png_path))

    # 4. Save default 4-type answer keys and per-question weights
    exam_cfg = create_default_exam_config(seed=2026)
    keys_json_path = keys_dir / "default_keys.json"
    exam_cfg.save_json(keys_json_path)

    print(f"[OK] PDF da Folha de Respostas gerado em: {pdf_path}")
    print(f"[OK] Layout canonico JSON salvo em:       {layout_json_path}")
    print(f"[OK] Imagem canonica de referencia em:    {ref_png_path}")
    print(f"[OK] Gabaritos (Tipos A-D) e pesos em:    {keys_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
