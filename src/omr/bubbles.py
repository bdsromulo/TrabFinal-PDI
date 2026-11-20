"""Stages 6, 7 & 8: Bubble grid ROI sampling, adaptive mark decision, and exam type reading.

For each bubble in the rectified sheet:
1. Samples an inner circular mask of radius `inner_radius_ratio * r` (default 0.75 * r) so that
   the printed outer ring of the circle is excluded from the fill measurement.
2. Computes the binary fill ratio `rho in [0, 1]` (fraction of foreground ink pixels inside the mask)
   and a grayscale darkness score.
3. Dynamically calibrates the decision threshold `tau` on the sheet's own distribution of 250 bubble
   fill ratios (separating the empty-bubble mode from the filled-bubble mode, with a safety floor
   `min_fill_threshold` so an all-blank sheet never produces false marks).
4. Classifies each question and the exam type block into:
   - `MARKED` (exactly 1 option >= threshold and clearly dominant)
   - `BLANK` (0 options >= threshold)
   - `MULTIPLE` (2 or more options >= threshold -> double mark / invalidated)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import cv2
import numpy as np

from .sheet import BubbleCoord, SheetLayout, get_default_layout

MarkStatus = Literal["MARKED", "BLANK", "MULTIPLE"]


@dataclass(frozen=True)
class BubbleReading:
    """Fill measurement for a single circular bubble ROI."""

    option: str
    cx_px: int
    cy_px: int
    radius_px: int
    fill_ratio: float
    mean_darkness: float
    is_marked: bool


@dataclass(frozen=True)
class QuestionReading:
    """Aggregated reading and decision for a question (or the exam-type block)."""

    index: int  # 0 for exam_type, 1..50 for questions
    status: MarkStatus
    marked_option: Optional[str]
    marked_options: Tuple[str, ...]
    fill_ratios: Dict[str, float]
    bubbles: Tuple[BubbleReading, ...]
    confidence: float = 1.0
    has_smudge_warning: bool = False


@dataclass
class SheetReading:
    """Complete reading of exam type and all 50 questions on a rectified sheet."""

    exam_type: QuestionReading
    questions: Dict[int, QuestionReading]
    calibrated_threshold: float
    empty_baseline_mean: float
    empty_baseline_std: float


def _build_circular_mask(radius_px: int) -> np.ndarray:
    """Creates a binary circular mask of shape (2*r+1, 2*r+1) with 1s inside radius_px."""
    size = 2 * radius_px + 1
    y, x = np.ogrid[-radius_px : radius_px + 1, -radius_px : radius_px + 1]
    mask = (x * x + y * y <= radius_px * radius_px).astype(np.uint8)
    return mask


def sample_bubble_fill(
    rectified_binary: np.ndarray,
    rectified_gray: Optional[np.ndarray],
    bubble: BubbleCoord,
    inner_radius_ratio: float = 0.74,
    max_micro_shift_px: int = 4,
) -> Tuple[float, float]:
    """Measures the binary fill ratio and normalized grayscale darkness inside a bubble's inner circle.

    Includes local micro-registration (`max_micro_shift_px` search around `(cx_px, cy_px)`)
    to compensate for mild radial lens barrel distortion, paper bowing, or off-center human marks.
    """
    h, w = rectified_binary.shape[:2]
    inner_r = max(3, int(round(bubble.radius_px * inner_radius_ratio)))

    mask = _build_circular_mask(inner_r)
    mask_area = float(np.count_nonzero(mask))
    if mask_area <= 0:
        return 0.0, 0.0

    def _eval_at(cx: int, cy: int) -> Tuple[float, float]:
        x0 = cx - inner_r
        y0 = cy - inner_r
        x1 = cx + inner_r + 1
        y1 = cy + inner_r + 1
        if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
            return 0.0, 0.0
        roi_bin = rectified_binary[y0:y1, x0:x1]
        active = float(np.count_nonzero((roi_bin > 0) & (mask > 0)))
        f_ratio = active / mask_area
        if rectified_gray is not None:
            roi_gray = rectified_gray[y0:y1, x0:x1]
            mean_val = float(np.mean(roi_gray[mask > 0]))
            dark = max(0.0, min(1.0, (255.0 - mean_val) / 255.0))
        else:
            dark = f_ratio
        return f_ratio, dark

    best_fill, best_dark = _eval_at(bubble.cx_px, bubble.cy_px)

    # If there is non-trivial ink near the center, check small shifts within max_micro_shift_px
    # while keeping the inner mask strictly inside the printed ring (shift + inner_r <= bubble.radius_px - 3)
    radial_clearance = bubble.radius_px - inner_r - 3
    safe_shift = min(max_micro_shift_px, max(0, radial_clearance))
    if safe_shift >= 1 and (best_fill > 0.06 or best_dark > 0.08):
        for dy in (-safe_shift, 0, safe_shift):
            for dx in (-safe_shift, 0, safe_shift):
                if dx == 0 and dy == 0:
                    continue
                if dx * dx + dy * dy > radial_clearance * radial_clearance:
                    continue
                f_cand, d_cand = _eval_at(bubble.cx_px + dx, bubble.cy_px + dy)
                if f_cand > best_fill:
                    best_fill, best_dark = f_cand, d_cand

    # Hybrid effective score: combines binary fill ratio with contrast darkness boost
    # so faint pencil marks or thin 'X' / checkmark strokes that darken the interior are reinforced
    effective_fill = float(max(best_fill, 0.72 * best_fill + 0.35 * max(0.0, best_dark - 0.08)))
    return min(1.0, effective_fill), float(best_dark)


def calibrate_mark_threshold(
    all_fill_ratios: Sequence[float],
    min_fill_threshold: float = 0.22,
    max_fill_threshold: float = 0.55,
) -> Tuple[float, float, float]:
    """Calibrates the fill-ratio threshold from the sheet's own bubble distribution.

    Since at least ~75-80% of all bubbles on a 5-option multiple-choice sheet are empty
    (even if every question has 1 mark), the lower 70th percentile reliably characterizes
    the empty-bubble baseline `(mu_empty, sigma_empty)`. When filled bubbles are present,
    we combine the baseline floor with the midpoint between the empty cluster and the
    filled cluster (1D Otsu / bimodal gap).

    Returns:
        Tuple `(threshold, empty_mean, empty_std)`.
    """
    vals = np.asarray(all_fill_ratios, dtype=np.float64)
    if vals.size == 0:
        return min_fill_threshold, 0.0, 0.0

    sorted_vals = np.sort(vals)
    # The lowest 65% of bubbles are guaranteed to be unmarked in any realistic exam sheet
    n_empty = max(1, int(round(0.65 * sorted_vals.size)))
    empty_sample = sorted_vals[:n_empty]
    empty_mean = float(np.mean(empty_sample))
    empty_std = float(np.std(empty_sample))

    statistical_floor = max(min_fill_threshold, empty_mean + 4.5 * max(0.015, empty_std))

    # Check if there is a high-fill cluster (marked bubbles)
    high_vals = sorted_vals[sorted_vals >= statistical_floor]
    if high_vals.size >= 3:
        filled_median = float(np.median(high_vals))
        # Midpoint biased slightly toward empty_mean to catch lighter pen marks while staying above noise
        bimodal_split = empty_mean + 0.42 * (filled_median - empty_mean)
        threshold = max(statistical_floor, bimodal_split)
    else:
        threshold = statistical_floor

    threshold = float(np.clip(threshold, min_fill_threshold, max_fill_threshold))
    return threshold, empty_mean, empty_std


def _decide_group(
    index: int,
    bubbles_coords: Sequence[BubbleCoord],
    raw_measurements: Sequence[Tuple[float, float]],
    threshold: float,
    relative_dominance_margin: float = 0.12,
) -> QuestionReading:
    """Decides MARKED, BLANK, or MULTIPLE for a group of bubbles, distinguishing smudges from double marks."""
    marked_opts: List[str] = []
    bubble_readings: List[BubbleReading] = []
    fill_map: Dict[str, float] = {}

    fills = [m[0] for m in raw_measurements]
    sorted_fills = sorted(fills, reverse=True)
    max_fill = sorted_fills[0] if sorted_fills else 0.0
    second_fill = sorted_fills[1] if len(sorted_fills) > 1 else 0.0

    # If the primary mark is very strong (e.g. >= 0.65) and a secondary option has a light erasure smudge
    # (just slightly above threshold, but < 0.42 * max_fill and < 0.36), treat the secondary as an erased smudge
    has_smudge_warning = False
    effective_second_cutoff = threshold
    if max_fill >= max(0.58, threshold * 1.6):
        smudge_ceiling = min(0.36, 0.42 * max_fill)
        if threshold <= second_fill <= smudge_ceiling:
            has_smudge_warning = True
            effective_second_cutoff = second_fill + 1e-4

    for b, (fill_ratio, darkness) in zip(bubbles_coords, raw_measurements):
        fill_map[b.option] = fill_ratio
        is_marked = fill_ratio >= effective_second_cutoff
        if not is_marked and max_fill >= threshold and fill_ratio >= max(0.20, threshold * 0.82):
            if (max_fill - fill_ratio) < relative_dominance_margin * max_fill:
                is_marked = True

        if is_marked:
            marked_opts.append(b.option)

        bubble_readings.append(
            BubbleReading(
                option=b.option,
                cx_px=b.cx_px,
                cy_px=b.cy_px,
                radius_px=b.radius_px,
                fill_ratio=fill_ratio,
                mean_darkness=darkness,
                is_marked=is_marked,
            )
        )

    if len(marked_opts) == 0:
        status: MarkStatus = "BLANK"
        single_opt = None
        confidence = float(np.clip(1.0 - max_fill / max(1e-4, threshold), 0.5, 1.0))
    elif len(marked_opts) == 1:
        status = "MARKED"
        single_opt = marked_opts[0]
        margin = (max_fill - second_fill) / max(1e-4, max_fill)
        confidence = float(np.clip(margin, 0.3, 1.0))
    else:
        status = "MULTIPLE"
        single_opt = None
        confidence = float(np.clip(second_fill / max(1e-4, max_fill), 0.5, 1.0))

    return QuestionReading(
        index=index,
        status=status,
        marked_option=single_opt,
        marked_options=tuple(marked_opts),
        fill_ratios=fill_map,
        bubbles=tuple(bubble_readings),
        confidence=round(confidence, 4),
        has_smudge_warning=has_smudge_warning,
    )


def read_sheet_bubbles(
    rectified_binary: np.ndarray,
    rectified_gray: Optional[np.ndarray] = None,
    layout: Optional[SheetLayout] = None,
    inner_radius_ratio: float = 0.74,
    min_fill_threshold: float = 0.22,
    fixed_threshold: Optional[float] = None,
) -> SheetReading:
    """Samples all 250 question bubbles and 4 exam-type bubbles and makes mark decisions."""
    if layout is None:
        layout = get_default_layout()

    # 1. Sample all question bubbles
    q_coords_by_num: Dict[int, List[BubbleCoord]] = {}
    q_meas_by_num: Dict[int, List[Tuple[float, float]]] = {}
    all_fills: List[float] = []

    for q_num in range(1, layout.num_questions + 1):
        coords = layout.get_question_bubbles(q_num)
        meas = [
            sample_bubble_fill(rectified_binary, rectified_gray, b, inner_radius_ratio)
            for b in coords
        ]
        q_coords_by_num[q_num] = coords
        q_meas_by_num[q_num] = meas
        all_fills.extend(m[0] for m in meas)

    # 2. Sample exam-type bubbles
    type_coords = layout.get_exam_type_bubbles()
    type_meas = [
        sample_bubble_fill(rectified_binary, rectified_gray, b, inner_radius_ratio)
        for b in type_coords
    ]
    all_fills.extend(m[0] for m in type_meas)

    # 3. Calibrate threshold dynamically on this sheet (unless fixed_threshold is explicitly set)
    cal_thresh, empty_mean, empty_std = calibrate_mark_threshold(
        all_fills,
        min_fill_threshold=min_fill_threshold,
    )
    threshold = float(fixed_threshold) if fixed_threshold is not None else cal_thresh

    # 4. Decide exam type and each of the 50 questions
    exam_type_reading = _decide_group(0, type_coords, type_meas, threshold)
    questions_reading: Dict[int, QuestionReading] = {}
    for q_num in range(1, layout.num_questions + 1):
        questions_reading[q_num] = _decide_group(
            q_num,
            q_coords_by_num[q_num],
            q_meas_by_num[q_num],
            threshold,
        )

    return SheetReading(
        exam_type=exam_type_reading,
        questions=questions_reading,
        calibrated_threshold=threshold,
        empty_baseline_mean=empty_mean,
        empty_baseline_std=empty_std,
    )
