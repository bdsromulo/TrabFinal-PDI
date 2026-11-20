"""Canonical geometrical specification of the OMR Answer Sheet (Folha de Respostas).

This module is the single source of truth for coordinates shared across:
1. The printable PDF generator (`tools/generate_sheet.py`)
2. The reference raster image renderer (`render_blank_sheet` / `render_filled_sheet`)
3. The homography rectification target (`src/omr/rectify.py`)
4. The ROI bubble sampler (`src/omr/bubbles.py`)

Coordinate system:
- Origin (0, 0) is the top-left corner of an A4 page (210 mm x 297 mm).
- X increases to the right [0 .. 210 mm]; Y increases downwards [0 .. 297 mm].
- Canonical scale is 8.0 pixels per millimeter (1680 x 2376 pixels, ~203.2 DPI),
  providing exact integer pixel coordinates for all markers and bubbles.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

OPTIONS: Tuple[str, ...] = ("A", "B", "C", "D", "E")
EXAM_TYPES: Tuple[str, ...] = ("A", "B", "C", "D")
MARKER_NAMES: Tuple[str, ...] = ("top_left", "top_right", "bottom_right", "bottom_left")


@dataclass(frozen=True)
class MarkerCoord:
    """Canonical specification of a square corner fiducial marker."""

    name: str
    cx_mm: float
    cy_mm: float
    size_mm: float
    px_per_mm: float = 8.0

    @property
    def cx_px(self) -> float:
        return self.cx_mm * self.px_per_mm

    @property
    def cy_px(self) -> float:
        return self.cy_mm * self.px_per_mm

    @property
    def size_px(self) -> float:
        return self.size_mm * self.px_per_mm

    @property
    def bbox_px(self) -> Tuple[int, int, int, int]:
        """Returns (x_min, y_min, x_max, y_max) in canonical pixels."""
        half = self.size_px / 2.0
        return (
            int(round(self.cx_px - half)),
            int(round(self.cy_px - half)),
            int(round(self.cx_px + half)),
            int(round(self.cy_px + half)),
        )


@dataclass(frozen=True)
class BubbleCoord:
    """Canonical specification of a circular answer or exam-type bubble."""

    group: str  # "exam_type" or "question"
    index: int  # 0 for exam_type, or 1..50 for questions
    option: str  # "A".."D" for exam_type, or "A".."E" for questions
    cx_mm: float
    cy_mm: float
    radius_mm: float
    px_per_mm: float = 8.0

    @property
    def cx_px(self) -> int:
        return int(round(self.cx_mm * self.px_per_mm))

    @property
    def cy_px(self) -> int:
        return int(round(self.cy_mm * self.px_per_mm))

    @property
    def radius_px(self) -> int:
        return int(round(self.radius_mm * self.px_per_mm))


@dataclass(frozen=True)
class SheetLayout:
    """Complete canonical geometry of the 50-question OMR answer sheet."""

    page_width_mm: float = 210.0
    page_height_mm: float = 297.0
    px_per_mm: float = 8.0
    num_questions: int = 50
    options: Tuple[str, ...] = OPTIONS
    exam_types: Tuple[str, ...] = EXAM_TYPES
    markers: Tuple[MarkerCoord, ...] = field(default_factory=tuple)
    exam_type_bubbles: Tuple[BubbleCoord, ...] = field(default_factory=tuple)
    question_bubbles: Tuple[BubbleCoord, ...] = field(default_factory=tuple)

    @property
    def canonical_width(self) -> int:
        return int(round(self.page_width_mm * self.px_per_mm))

    @property
    def canonical_height(self) -> int:
        return int(round(self.page_height_mm * self.px_per_mm))

    @property
    def canonical_size(self) -> Tuple[int, int]:
        """Returns (width, height) in pixels for OpenCV warpPerspective."""
        return (self.canonical_width, self.canonical_height)

    @property
    def marker_centers_px(self) -> np.ndarray:
        """Returns 4x2 float32 array of canonical marker centers in order [TL, TR, BR, BL]."""
        return np.array(
            [[m.cx_px, m.cy_px] for m in self.markers],
            dtype=np.float32,
        )

    def get_question_bubbles(self, question_number: int) -> List[BubbleCoord]:
        """Returns the 5 bubble coordinates (A..E) for a 1-based question number."""
        return [b for b in self.question_bubbles if b.index == question_number]

    def get_exam_type_bubbles(self) -> List[BubbleCoord]:
        """Returns the 4 exam-type bubble coordinates (A..D)."""
        return list(self.exam_type_bubbles)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def get_default_layout(px_per_mm: float = 8.0) -> SheetLayout:
    """Constructs the canonical UTFPR PDI 50-question, 4-exam-type answer sheet layout."""
    marker_size_mm = 14.0
    markers = (
        MarkerCoord("top_left", cx_mm=18.0, cy_mm=18.0, size_mm=marker_size_mm, px_per_mm=px_per_mm),
        MarkerCoord("top_right", cx_mm=192.0, cy_mm=18.0, size_mm=marker_size_mm, px_per_mm=px_per_mm),
        MarkerCoord("bottom_right", cx_mm=192.0, cy_mm=279.0, size_mm=marker_size_mm, px_per_mm=px_per_mm),
        MarkerCoord("bottom_left", cx_mm=18.0, cy_mm=279.0, size_mm=marker_size_mm, px_per_mm=px_per_mm),
    )

    # Exam type bubbles (A, B, C, D) in top-right header box
    type_y_mm = 44.0
    type_x_start_mm = 138.0
    type_dx_mm = 11.0
    type_radius_mm = 3.0
    exam_type_bubbles = tuple(
        BubbleCoord(
            group="exam_type",
            index=0,
            option=opt,
            cx_mm=type_x_start_mm + i * type_dx_mm,
            cy_mm=type_y_mm,
            radius_mm=type_radius_mm,
            px_per_mm=px_per_mm,
        )
        for i, opt in enumerate(EXAM_TYPES)
    )

    # 50 Question bubbles split into 2 columns of 25 rows each
    # Column 0: Questions 1..25; Column 1: Questions 26..50
    col_x_starts_mm = (46.0, 132.0)
    opt_dx_mm = 9.0
    row_y_start_mm = 70.0
    row_dy_mm = 8.0
    q_radius_mm = 2.75

    q_bubbles: List[BubbleCoord] = []
    for q_num in range(1, 51):
        col_idx = 0 if q_num <= 25 else 1
        row_idx = (q_num - 1) if q_num <= 25 else (q_num - 26)
        cy_mm = row_y_start_mm + row_idx * row_dy_mm
        x0_mm = col_x_starts_mm[col_idx]
        for opt_idx, opt in enumerate(OPTIONS):
            cx_mm = x0_mm + opt_idx * opt_dx_mm
            q_bubbles.append(
                BubbleCoord(
                    group="question",
                    index=q_num,
                    option=opt,
                    cx_mm=cx_mm,
                    cy_mm=cy_mm,
                    radius_mm=q_radius_mm,
                    px_per_mm=px_per_mm,
                )
            )

    return SheetLayout(
        page_width_mm=210.0,
        page_height_mm=297.0,
        px_per_mm=px_per_mm,
        num_questions=50,
        options=OPTIONS,
        exam_types=EXAM_TYPES,
        markers=markers,
        exam_type_bubbles=exam_type_bubbles,
        question_bubbles=tuple(q_bubbles),
    )


def create_custom_layout(
    num_questions: int = 50,
    num_options: int = 5,
    num_columns: Optional[int] = None,
    exam_types: Sequence[str] = EXAM_TYPES,
    px_per_mm: float = 8.0,
) -> SheetLayout:
    """Builds a parameterized A4 OMR SheetLayout for arbitrary exam configurations.

    Supports from 5 up to 100 questions, 2 to 5 options per question (e.g. V/F, A-D, A-E),
    and 1 to 4 columns while preserving the 4 canonical corner fiducial markers.
    """
    if num_questions < 1 or num_questions > 120:
        raise ValueError("num_questions deve estar entre 1 e 120.")
    if num_options < 2 or num_options > 5:
        raise ValueError("num_options deve estar entre 2 e 5.")

    if num_questions == 50 and num_options == 5 and (num_columns in (None, 2)) and tuple(exam_types) == EXAM_TYPES:
        return get_default_layout(px_per_mm=px_per_mm)

    opts = OPTIONS[:num_options]
    types_tuple = tuple(str(t).upper() for t in exam_types)

    if num_columns is None:
        if num_questions <= 25:
            num_columns = 1
        elif num_questions <= 50:
            num_columns = 2
        elif num_questions <= 75:
            num_columns = 3
        else:
            num_columns = 4

    rows_per_col = int(np.ceil(num_questions / float(num_columns)))

    marker_size_mm = 14.0
    markers = (
        MarkerCoord("top_left", cx_mm=18.0, cy_mm=18.0, size_mm=marker_size_mm, px_per_mm=px_per_mm),
        MarkerCoord("top_right", cx_mm=192.0, cy_mm=18.0, size_mm=marker_size_mm, px_per_mm=px_per_mm),
        MarkerCoord("bottom_right", cx_mm=192.0, cy_mm=279.0, size_mm=marker_size_mm, px_per_mm=px_per_mm),
        MarkerCoord("bottom_left", cx_mm=18.0, cy_mm=279.0, size_mm=marker_size_mm, px_per_mm=px_per_mm),
    )

    type_y_mm = 44.0
    type_x_start_mm = 138.0
    type_dx_mm = min(11.0, 40.0 / max(1, len(types_tuple) - 1))
    type_radius_mm = 3.0
    exam_type_bubbles = tuple(
        BubbleCoord(
            group="exam_type",
            index=0,
            option=opt,
            cx_mm=type_x_start_mm + i * type_dx_mm,
            cy_mm=type_y_mm,
            radius_mm=type_radius_mm,
            px_per_mm=px_per_mm,
        )
        for i, opt in enumerate(types_tuple)
    )

    # Compute horizontal and vertical geometry to fit comfortably inside [28..184] x [68..264] mm
    usable_w_mm = 154.0
    col_stride_mm = usable_w_mm / float(num_columns)
    opt_dx_mm = min(9.0, (col_stride_mm - 14.0) / max(1, num_options - 1))
    row_y_start_mm = 70.0
    usable_h_mm = 192.0
    row_dy_mm = min(8.0, usable_h_mm / max(1, rows_per_col - 1))
    q_radius_mm = min(2.75, 0.34 * min(opt_dx_mm, row_dy_mm))

    q_bubbles: List[BubbleCoord] = []
    for q_num in range(1, num_questions + 1):
        col_idx = (q_num - 1) // rows_per_col
        row_idx = (q_num - 1) % rows_per_col
        cy_mm = row_y_start_mm + row_idx * row_dy_mm
        x0_mm = 38.0 + col_idx * col_stride_mm
        for opt_idx, opt in enumerate(opts):
            cx_mm = x0_mm + opt_idx * opt_dx_mm
            q_bubbles.append(
                BubbleCoord(
                    group="question",
                    index=q_num,
                    option=opt,
                    cx_mm=cx_mm,
                    cy_mm=cy_mm,
                    radius_mm=q_radius_mm,
                    px_per_mm=px_per_mm,
                )
            )

    return SheetLayout(
        page_width_mm=210.0,
        page_height_mm=297.0,
        px_per_mm=px_per_mm,
        num_questions=num_questions,
        options=opts,
        exam_types=types_tuple,
        markers=markers,
        exam_type_bubbles=exam_type_bubbles,
        question_bubbles=tuple(q_bubbles),
    )


def render_sheet_image(
    layout: Optional[SheetLayout] = None,
    exam_type_marked: Optional[str | Sequence[str]] = None,
    marked_answers: Optional[Mapping[int, str | Sequence[str]]] = None,
    mark_intensity: int = 28,
    mark_color_bgr: Optional[Tuple[int, int, int]] = None,
    mark_style: str = "solid",  # "solid", "cross", "check", "partial"
    fill_Jitter_px: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Renders a canonical high-resolution BGR image of the answer sheet (blank or filled).

    Args:
        layout: SheetLayout specification (defaults to `get_default_layout()`).
        exam_type_marked: Optional exam type ("A".."D") or sequence of types to fill.
        marked_answers: Optional mapping `{question_number: "A" | ["A", "C"] | ""}`.
        mark_intensity: Grayscale intensity [0..255] of pen marks (default 28 = dark ink).
        fill_Jitter_px: Optional random jitter in pixels to simulate human hand imperfection.
        rng: NumPy random generator for reproducible jitter/texture.

    Returns:
        Canonical BGR uint8 image of shape (canonical_height, canonical_width, 3).
    """
    if layout is None:
        layout = get_default_layout()
    if rng is None:
        rng = np.random.default_rng(42)

    w, h = layout.canonical_width, layout.canonical_height
    px = layout.px_per_mm
    img = np.full((h, w, 3), 255, dtype=np.uint8)

    # 1. Draw the 4 solid black corner fiducial squares
    for m in layout.markers:
        x0, y0, x1, y1 = m.bbox_px
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), thickness=-1)

    # 2. Header text & boxes (staying strictly inside x in [32, 178] mm, y in [15, 55] mm)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        img,
        "UTFPR - IF69D / OPEET014 - FOLHA DE RESPOSTAS (OMR)",
        (int(33 * px), int(20 * px)),
        font,
        0.75 * (px / 8.0),
        (30, 30, 30),
        max(1, int(round(2 * px / 8.0))),
        cv2.LINE_AA,
    )

    # Student identification box (thin border)
    id_x0, id_y0 = int(32 * px), int(27 * px)
    id_x1, id_y1 = int(124 * px), int(52 * px)
    cv2.rectangle(img, (id_x0, id_y0), (id_x1, id_y1), (120, 120, 120), max(1, int(px / 4)))
    cv2.putText(
        img,
        "ALUNO(A): ____________________________________",
        (int(35 * px), int(36 * px)),
        font,
        0.55 * (px / 8.0),
        (60, 60, 60),
        max(1, int(px / 5)),
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        "RA: ___________________   DATA: ____/____/2026",
        (int(35 * px), int(47 * px)),
        font,
        0.55 * (px / 8.0),
        (60, 60, 60),
        max(1, int(px / 5)),
        cv2.LINE_AA,
    )

    # Exam Type box
    tp_x0, tp_y0 = int(128 * px), int(27 * px)
    tp_x1, tp_y1 = int(180 * px), int(52 * px)
    cv2.rectangle(img, (tp_x0, tp_y0), (tp_x1, tp_y1), (120, 120, 120), max(1, int(px / 4)))
    cv2.putText(
        img,
        "TIPO DE PROVA",
        (int(138 * px), int(33 * px)),
        font,
        0.55 * (px / 8.0),
        (40, 40, 40),
        max(1, int(round(1.5 * px / 8.0))),
        cv2.LINE_AA,
    )

    ring_thickness = max(1, int(round(1.6 * px / 8.0)))
    for b in layout.exam_type_bubbles:
        # Label directly above each exam-type bubble
        cv2.putText(
            img,
            b.option,
            (b.cx_px - int(1.3 * px), b.cy_px - b.radius_px - int(1.2 * px)),
            font,
            0.55 * (px / 8.0),
            (40, 40, 40),
            max(1, int(round(1.5 * px / 8.0))),
            cv2.LINE_AA,
        )
        cv2.circle(img, (b.cx_px, b.cy_px), b.radius_px, (50, 50, 50), ring_thickness, cv2.LINE_AA)

    # 3. Column headers and Question numbers/bubbles (adaptive to any custom column/bubble size)
    header_y_px = int(63.0 * px)
    seen_col_x0: List[int] = []
    for q_num in range(1, layout.num_questions + 1):
        q_bubbles = layout.get_question_bubbles(q_num)
        first_b = q_bubbles[0]
        r_scale = first_b.radius_mm / 2.75

        if first_b.cx_px not in seen_col_x0:
            seen_col_x0.append(first_b.cx_px)
            q_hdr_x = first_b.cx_px - int(round(12.0 * r_scale * px))
            cv2.putText(
                img,
                "Q.",
                (q_hdr_x, header_y_px),
                font,
                0.55 * r_scale * (px / 8.0),
                (40, 40, 40),
                max(1, int(round(1.6 * r_scale * px / 8.0))),
                cv2.LINE_AA,
            )
            for b in q_bubbles:
                cv2.putText(
                    img,
                    b.option,
                    (b.cx_px - int(round(1.4 * r_scale * px)), header_y_px),
                    font,
                    0.60 * r_scale * (px / 8.0),
                    (30, 30, 30),
                    max(1, int(round(1.8 * r_scale * px / 8.0))),
                    cv2.LINE_AA,
                )

        label_x = first_b.cx_px - int(round(12.5 * r_scale * px))
        label_y = first_b.cy_px + int(round(1.6 * r_scale * px))
        cv2.putText(
            img,
            f"{q_num:02d}",
            (label_x, label_y),
            font,
            0.55 * r_scale * (px / 8.0),
            (40, 40, 40),
            max(1, int(round(1.5 * r_scale * px / 8.0))),
            cv2.LINE_AA,
        )

        q_ring_thick = max(1, int(round(1.6 * r_scale * px / 8.0)))
        for b in q_bubbles:
            cv2.circle(img, (b.cx_px, b.cy_px), b.radius_px, (55, 55, 55), q_ring_thick, cv2.LINE_AA)

    # 5. Apply simulated pen markings if provided
    def _fill_bubble(b: BubbleCoord) -> None:
        dx = int(rng.integers(-fill_Jitter_px, fill_Jitter_px + 1)) if fill_Jitter_px > 0 else 0
        dy = int(rng.integers(-fill_Jitter_px, fill_Jitter_px + 1)) if fill_Jitter_px > 0 else 0
        cx_i = b.cx_px + dx
        cy_i = b.cy_px + dy
        color = (
            mark_color_bgr
            if mark_color_bgr is not None
            else (int(mark_intensity), int(mark_intensity), int(mark_intensity))
        )
        if mark_style == "solid":
            fill_r = max(2, int(round(b.radius_px * 0.86)))
            cv2.circle(img, (cx_i, cy_i), fill_r, color, thickness=-1, lineType=cv2.LINE_AA)
        elif mark_style == "partial":
            fill_r = max(2, int(round(b.radius_px * 0.62)))
            cv2.circle(img, (cx_i, cy_i), fill_r, color, thickness=-1, lineType=cv2.LINE_AA)
        elif mark_style == "cross":
            arm = max(3, int(round(b.radius_px * 0.72)))
            thick = max(3, int(round(b.radius_px * 0.44)))
            cv2.line(img, (cx_i - arm, cy_i - arm), (cx_i + arm, cy_i + arm), color, thick, cv2.LINE_AA)
            cv2.line(img, (cx_i - arm, cy_i + arm), (cx_i + arm, cy_i - arm), color, thick, cv2.LINE_AA)
        elif mark_style == "check":
            arm = max(3, int(round(b.radius_px * 0.72)))
            thick = max(3, int(round(b.radius_px * 0.46)))
            p1 = (cx_i - int(arm * 0.8), cy_i)
            p2 = (cx_i - int(arm * 0.1), cy_i + int(arm * 0.75))
            p3 = (cx_i + int(arm * 0.9), cy_i - int(arm * 0.75))
            cv2.line(img, p1, p2, color, thick, cv2.LINE_AA)
            cv2.line(img, p2, p3, color, thick, cv2.LINE_AA)
        else:
            raise ValueError(f"Unknown mark_style: {mark_style}")

    if exam_type_marked:
        types_to_mark = (
            [exam_type_marked] if isinstance(exam_type_marked, str) else list(exam_type_marked)
        )
        for b in layout.exam_type_bubbles:
            if b.option in types_to_mark:
                _fill_bubble(b)

    if marked_answers:
        for q_num, ans in marked_answers.items():
            if not ans:
                continue
            opts_to_mark = [ans] if isinstance(ans, str) else list(ans)
            for b in layout.get_question_bubbles(int(q_num)):
                if b.option in opts_to_mark:
                    _fill_bubble(b)

    return img
