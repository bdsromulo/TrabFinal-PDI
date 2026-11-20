"""Stage 5: Projective rectification via Homography (Week 2 — Geometric & Projective Transformations).

Estimates the 3x3 projective transformation matrix H that maps the 4 detected fiducial marker
centroids in the smartphone photo to their canonical positions in `SheetLayout`, validates the
geometric conditioning of H, and warps the sheet into the canonical coordinate frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .preprocess import BinarizeMethod, binarize_image, normalize_illumination_background
from .sheet import SheetLayout, get_default_layout


class RectificationError(RuntimeError):
    """Raised when the estimated homography is degenerate or geometrically implausible."""


@dataclass
class RectificationResult:
    """Output of the projective rectification stage."""

    homography: np.ndarray  # 3x3 float64 matrix
    rectified_color: np.ndarray  # BGR canonical image
    rectified_gray: np.ndarray  # Grayscale canonical image
    rectified_normalized: np.ndarray  # Illumination-normalized canonical grayscale
    rectified_binary: np.ndarray  # Clean binarized canonical image (ink = 255)
    grid_overlay: np.ndarray  # Debug image showing canonical bubble ROIs over rectified sheet


def validate_homography(homography: Optional[np.ndarray], src_corners: np.ndarray) -> None:
    """Verifies that the 3x3 homography matrix is invertible and well-conditioned."""
    if homography is None or homography.shape != (3, 3):
        raise RectificationError("Falha ao estimar a matriz de homografia 3x3.")

    if not np.all(np.isfinite(homography)):
        raise RectificationError("A matriz de homografia contem valores nao finitos.")

    det = float(np.linalg.det(homography))
    if abs(det) < 1e-9:
        raise RectificationError(f"Homografia degenerada (determinante muito proximo de zero: {det:.3e}).")

    # Check perspective tilt terms H[2,0] and H[2,1]
    h_norm = homography / (homography[2, 2] if abs(homography[2, 2]) > 1e-12 else 1.0)
    persp_mag = float(np.hypot(h_norm[2, 0], h_norm[2, 1]))
    if persp_mag > 0.015:
        raise RectificationError(
            f"Distorcao de perspectiva extrema detectada (magnitude projetiva={persp_mag:.4f}). "
            f"Fotografe a folha com menor inclinacao."
        )


def draw_canonical_grid_overlay(
    rectified_bgr: np.ndarray,
    layout: SheetLayout,
) -> np.ndarray:
    """Draws the canonical marker boxes, exam type bubbles, and 50x5 question bubble ROIs."""
    overlay = rectified_bgr.copy()
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)

    for m in layout.markers:
        x0, y0, x1, y1 = m.bbox_px
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 200, 0), 2)
        cv2.circle(overlay, (int(round(m.cx_px)), int(round(m.cy_px))), 4, (0, 0, 255), -1)

    for b in layout.exam_type_bubbles:
        cv2.circle(overlay, (b.cx_px, b.cy_px), b.radius_px, (255, 140, 0), 2, cv2.LINE_AA)

    for b in layout.question_bubbles:
        cv2.circle(overlay, (b.cx_px, b.cy_px), b.radius_px, (230, 120, 0), 1, cv2.LINE_AA)

    return overlay


def _disambiguate_cyclic_orientation(
    gray_src: np.ndarray,
    src_pts: np.ndarray,
    layout: Optional[SheetLayout] = None,
) -> np.ndarray:
    """Resolves 360-degree orientation ambiguity (0, 90, 180, 270 deg) across the 4 cyclic corner shifts.

    Uses a shadow-invariant local adaptive binary preview at 1/4 scale. In the true upright (0 deg)
    orientation, the top header strip (`y in [16..36] mm, x in [35..175] mm`) contains the printed
    header title and top borders of the ID/Exam Type boxes, whereas the three other cyclic shifts
    place either the empty bottom margin (180 deg) or the empty left/right lateral margins (90/270 deg)
    into that top strip.
    """
    if layout is None:
        layout = get_default_layout()

    dst_pts = layout.marker_centers_px
    scale = 0.25
    small_w = int(round(layout.canonical_width * scale))
    small_h = int(round(layout.canonical_height * scale))
    dst_small = dst_pts * scale
    px_s = layout.px_per_mm * scale

    best_shift = 0
    best_score = -1e9

    for shift in range(4):
        cand_pts = np.roll(src_pts, -shift, axis=0)

        H_small, _ = cv2.findHomography(cand_pts, dst_small, method=0)
        if H_small is None:
            continue

        preview_gray = cv2.warpPerspective(
            gray_src,
            H_small,
            (small_w, small_h),
            flags=cv2.INTER_AREA,
            borderValue=255,
        )
        # Shadow-invariant local adaptive binarization on the preview
        preview_bin = cv2.adaptiveThreshold(
            preview_gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=21,
            C=15,
        )

        # Top header strip (y: 16..36 mm, x: 34..176 mm) vs empty bottom strip (y: 265..281 mm, x: 34..176 mm)
        top_strip = preview_bin[int(16 * px_s) : int(36 * px_s), int(34 * px_s) : int(176 * px_s)]
        bot_strip = preview_bin[int(265 * px_s) : int(281 * px_s), int(34 * px_s) : int(176 * px_s)]

        top_ink = float(np.mean(top_strip > 0))
        bot_ink = float(np.mean(bot_strip > 0))

        # Also check the ID/Exam-Type box band (y: 36..52 mm, x: 34..176 mm)
        box_strip = preview_bin[int(36 * px_s) : int(52 * px_s), int(34 * px_s) : int(176 * px_s)]
        box_ink = float(np.mean(box_strip > 0))

        score = (top_ink + 0.5 * box_ink) - 1.5 * bot_ink
        if score > best_score:
            best_score = score
            best_shift = shift

    return np.roll(src_pts, -best_shift, axis=0)


def rectify_sheet(
    image: np.ndarray,
    src_corners_px: np.ndarray,
    layout: Optional[SheetLayout] = None,
    binarize_method: BinarizeMethod = "adaptive",
    normalize_illumination: bool = True,
    adaptive_block_size: int = 75,
    adaptive_c: int = 18,
    auto_orient_360: bool = True,
) -> RectificationResult:
    """Warps the input photo to the canonical answer sheet coordinate system via Homography."""
    if layout is None:
        layout = get_default_layout()

    if image.ndim == 2:
        color_src = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        gray_src = image
    else:
        color_src = image
        from .preprocess import to_grayscale
        gray_src = to_grayscale(image, chromatic_boost=True)

    src_pts = np.asarray(src_corners_px, dtype=np.float32).reshape(4, 2)
    if auto_orient_360:
        src_pts = _disambiguate_cyclic_orientation(gray_src, src_pts, layout)

    dst_pts = layout.marker_centers_px

    homography, _ = cv2.findHomography(src_pts, dst_pts, method=0)
    validate_homography(homography, src_pts)

    dsize = layout.canonical_size  # (width, height)

    rectified_color = cv2.warpPerspective(
        color_src,
        homography,
        dsize,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    rectified_gray = cv2.warpPerspective(
        gray_src,
        homography,
        dsize,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )

    if normalize_illumination:
        rectified_norm = normalize_illumination_background(rectified_gray, kernel_fraction=0.04)
    else:
        rectified_norm = rectified_gray.copy()

    target_gray = (
        rectified_gray
        if (binarize_method == "otsu" and not normalize_illumination)
        else rectified_norm
    )

    rectified_binary, _ = binarize_image(
        target_gray,
        method=binarize_method,
        adaptive_block_size=adaptive_block_size,
        adaptive_c=adaptive_c,
        morph_clean_ksize=0,
    )

    grid_overlay = draw_canonical_grid_overlay(rectified_color, layout)

    return RectificationResult(
        homography=homography,
        rectified_color=rectified_color,
        rectified_gray=rectified_gray,
        rectified_normalized=rectified_norm,
        rectified_binary=rectified_binary,
        grid_overlay=grid_overlay,
    )
