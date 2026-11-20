"""Stage 4: Corner fiducial marker detection using mathematical morphology and connected components.

Locates the 4 solid black square fiducial markers printed at the corners of the answer sheet:
1. Multi-level contour extraction (`cv2.RETR_LIST`) and closed-contour hole filling so markers inside
   a page boundary are never shadowed by an outer tabletop-to-paper edge contour.
2. Rotation-invariant geometric filtering via `cv2.minAreaRect` (rotated bounding box extent and aspect
   ratio), convex hull solidity, and polygon approximation.
3. Selecting the best 4 corner markers forming a valid convex quadrilateral at any 0..360 deg orientation
   ordered in clockwise cyclic order around their centroid.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .sheet import MARKER_NAMES


class MarkerDetectionError(RuntimeError):
    """Raised when the 4 corner fiducial markers cannot be reliably detected."""


@dataclass(frozen=True)
class DetectedMarker:
    """Properties of a detected fiducial corner marker."""

    name: str
    cx: float
    cy: float
    area: float
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    aspect_ratio: float
    solidity: float
    score: float


@dataclass
class MarkerDetectionResult:
    """Result of the corner marker detection stage."""

    markers: Tuple[DetectedMarker, DetectedMarker, DetectedMarker, DetectedMarker]
    corners_px: np.ndarray  # 4x2 float32 in clockwise cyclic order
    candidates_count: int
    morph_image: np.ndarray
    debug_image: np.ndarray


def _polygon_area(pts: np.ndarray) -> float:
    """Computes the area of a 2D polygon using the Shoelace formula."""
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def _is_convex_quad(pts: np.ndarray) -> bool:
    """Checks if 4 ordered points form a strictly convex clockwise quadrilateral."""
    if pts.shape != (4, 2):
        return False
    cross_signs = []
    for i in range(4):
        p0 = pts[i]
        p1 = pts[(i + 1) % 4]
        p2 = pts[(i + 2) % 4]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = float(v1[0] * v2[1] - v1[1] * v2[0])
        cross_signs.append(cross)
    return all(c > 1e-3 for c in cross_signs)


def _extract_candidates(
    binary_img: np.ndarray,
    min_area_ratio: float = 0.00055,
    max_area_ratio: float = 0.025,
) -> Tuple[List[DetectedMarker], np.ndarray]:
    """Extracts square-like connected component candidates from a binary image (360-deg rotation invariant)."""
    h, w = binary_img.shape[:2]
    img_area = float(h * w)
    min_area = img_area * min_area_ratio
    max_area = img_area * max_area_ratio

    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel_close)

    # Use RETR_LIST so contours nested inside a full-page border contour are always examined
    all_contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(binary_img)
    border_margin = 3
    for cnt in all_contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if (
            x <= border_margin
            or y <= border_margin
            or (x + bw) >= (w - border_margin)
            or (y + bh) >= (h - border_margin)
        ):
            continue
        c_area = float(cv2.contourArea(cnt))
        if c_area < min_area * 0.65 or c_area > max_area * 1.5:
            continue
        aspect = float(bw) / float(max(1, bh))
        if 0.40 <= aspect <= 2.50:
            cv2.drawContours(filled, [cnt], -1, 255, thickness=-1)

    k_open = max(5, int(round(min(h, w) * 0.0045)))
    if k_open % 2 == 0:
        k_open += 1
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (k_open, k_open))
    morph = cv2.morphologyEx(filled, cv2.MORPH_OPEN, kernel_open)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(morph, connectivity=8)

    candidates: List[DetectedMarker] = []
    for label_idx in range(1, num_labels):
        x = int(stats[label_idx, cv2.CC_STAT_LEFT])
        y = int(stats[label_idx, cv2.CC_STAT_TOP])
        bw = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        bh = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        area = float(stats[label_idx, cv2.CC_STAT_AREA])

        if area < min_area or area > max_area:
            continue
        if bw < 12 or bh < 12:
            continue

        roi_mask = (labels[y : y + bh, x : x + bw] == label_idx).astype(np.uint8) * 255
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)

        # Rotation-invariant bounding box metrics via minAreaRect
        (_, _), (rw, rh), _ = cv2.minAreaRect(cnt)
        if rw < 10 or rh < 10:
            continue
        rot_aspect = float(max(rw, rh)) / float(max(1e-3, min(rw, rh)))
        if rot_aspect > 2.10:
            continue
        rot_extent = area / float(max(1.0, rw * rh))
        if rot_extent < 0.62:
            continue

        hull = cv2.convexHull(cnt)
        hull_area = float(cv2.contourArea(hull))
        if hull_area <= 0:
            continue
        solidity = area / hull_area
        if solidity < 0.80:
            continue

        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.045 * peri, True)
        poly_bonus = 1.20 if (len(approx) == 4) else (1.05 if (4 <= len(approx) <= 6) else 0.80)

        cx, cy = float(centroids[label_idx, 0]), float(centroids[label_idx, 1])

        aspect_score = 1.0 - min(0.8, abs(np.log(rot_aspect)))
        score = float(aspect_score * (solidity ** 2) * (0.35 + 0.65 * rot_extent) * poly_bonus)

        candidates.append(
            DetectedMarker(
                name="candidate",
                cx=cx,
                cy=cy,
                area=area,
                bbox=(x, y, bw, bh),
                aspect_ratio=rot_aspect,
                solidity=solidity,
                score=score,
            )
        )

    return candidates, morph


def _order_quad_cyclic_clockwise(
    markers: Sequence[DetectedMarker],
) -> Tuple[DetectedMarker, DetectedMarker, DetectedMarker, DetectedMarker]:
    """Orders 4 markers in clockwise cyclic order around their centroid."""
    cx_mean = float(np.mean([m.cx for m in markers]))
    cy_mean = float(np.mean([m.cy for m in markers]))

    cyclic = sorted(
        markers,
        key=lambda m: float(np.arctan2(m.cy - cy_mean, m.cx - cx_mean)),
    )

    start_idx = min(range(4), key=lambda i: cyclic[i].cx + cyclic[i].cy)
    ordered = [cyclic[(start_idx + k) % 4] for k in range(4)]
    return (ordered[0], ordered[1], ordered[2], ordered[3])


def _select_four_corners(
    candidates: Sequence[DetectedMarker],
    img_w: int,
    img_h: int,
) -> Optional[Tuple[DetectedMarker, DetectedMarker, DetectedMarker, DetectedMarker]]:
    """Selects the best 4 corner markers forming a valid convex quadrilateral at any 0..360 deg orientation."""
    if len(candidates) < 4:
        return None

    pool = sorted(candidates, key=lambda c: c.score, reverse=True)[:12]
    img_area = float(img_w * img_h)

    best_quad: Optional[Tuple[DetectedMarker, DetectedMarker, DetectedMarker, DetectedMarker]] = None
    best_total_score = -1e9

    for comb in combinations(pool, 4):
        ordered = _order_quad_cyclic_clockwise(comb)
        pts = np.array([[m.cx, m.cy] for m in ordered], dtype=np.float32)

        if not _is_convex_quad(pts):
            continue

        quad_area = _polygon_area(pts)
        if quad_area < 0.10 * img_area:
            continue

        areas = np.array([m.area for m in ordered], dtype=np.float64)
        area_ratio = float(areas.max() / max(1.0, areas.min()))
        if area_ratio > 5.5:
            continue

        edges = [float(np.linalg.norm(pts[(i + 1) % 4] - pts[i])) for i in range(4)]
        if min(edges) < 0.18 * min(img_w, img_h):
            continue

        opp_ratio_1 = max(edges[0], edges[2]) / max(1.0, min(edges[0], edges[2]))
        opp_ratio_2 = max(edges[1], edges[3]) / max(1.0, min(edges[1], edges[3]))
        if opp_ratio_1 > 2.25 or opp_ratio_2 > 2.25:
            continue

        mean_pair_a = 0.5 * (edges[0] + edges[2])
        mean_pair_b = 0.5 * (edges[1] + edges[3])
        rect_aspect = max(mean_pair_a, mean_pair_b) / max(1.0, min(mean_pair_a, mean_pair_b))
        if rect_aspect < 1.10 or rect_aspect > 2.15:
            continue

        aspect_bonus = 1.0 - min(0.5, abs(rect_aspect - 1.50) * 0.6)
        area_fraction_bonus = min(1.0, (quad_area / img_area) * 1.8)
        area_consistency = 1.0 / (1.0 + 0.20 * (area_ratio - 1.0))
        quality_sum = sum(m.score for m in ordered)
        total_score = quality_sum * area_consistency * aspect_bonus + 1.5 * area_fraction_bonus

        if total_score > best_total_score:
            best_total_score = total_score
            best_quad = (
                DetectedMarker(
                    MARKER_NAMES[0], ordered[0].cx, ordered[0].cy, ordered[0].area, ordered[0].bbox, ordered[0].aspect_ratio, ordered[0].solidity, ordered[0].score
                ),
                DetectedMarker(
                    MARKER_NAMES[1], ordered[1].cx, ordered[1].cy, ordered[1].area, ordered[1].bbox, ordered[1].aspect_ratio, ordered[1].solidity, ordered[1].score
                ),
                DetectedMarker(
                    MARKER_NAMES[2], ordered[2].cx, ordered[2].cy, ordered[2].area, ordered[2].bbox, ordered[2].aspect_ratio, ordered[2].solidity, ordered[2].score
                ),
                DetectedMarker(
                    MARKER_NAMES[3], ordered[3].cx, ordered[3].cy, ordered[3].area, ordered[3].bbox, ordered[3].aspect_ratio, ordered[3].solidity, ordered[3].score
                ),
            )

    return best_quad


def detect_corner_markers(
    binary_image: np.ndarray,
    gray_image: Optional[np.ndarray] = None,
    base_color_image: Optional[np.ndarray] = None,
) -> MarkerDetectionResult:
    """Detects the 4 corner fiducial markers from a binary image (with fallback to illumination-normalized Otsu)."""
    h, w = binary_image.shape[:2]
    candidates, morph = _extract_candidates(binary_image)
    selected = _select_four_corners(candidates, w, h)

    if selected is None and gray_image is not None:
        from .preprocess import normalize_illumination_background

        norm = normalize_illumination_background(gray_image, kernel_fraction=0.095)
        _, fallback_bin = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        fb_candidates, fb_morph = _extract_candidates(fallback_bin)
        fb_selected = _select_four_corners(fb_candidates, w, h)
        if fb_selected is not None:
            candidates = fb_candidates
            morph = fb_morph
            selected = fb_selected

    if selected is None:
        raise MarkerDetectionError(
            f"Nao foi possivel detectar os 4 marcadores fiduciais de canto "
            f"(candidatos quadrados encontrados: {len(candidates)}). "
            f"Verifique se os 4 cantos da folha estao visiveis e bem iluminados."
        )

    corners_px = np.array([[m.cx, m.cy] for m in selected], dtype=np.float32)

    if gray_image is not None:
        from .rectify import _disambiguate_cyclic_orientation

        oriented_pts = _disambiguate_cyclic_orientation(gray_image, corners_px)
        # Find which cyclic shift was applied so we rotate `selected` markers to match [TL, TR, BR, BL]
        for shift in range(4):
            if np.allclose(np.roll(corners_px, -shift, axis=0), oriented_pts):
                ordered_list = [selected[(shift + k) % 4] for k in range(4)]
                selected = tuple(
                    DetectedMarker(
                        MARKER_NAMES[k],
                        ordered_list[k].cx,
                        ordered_list[k].cy,
                        ordered_list[k].area,
                        ordered_list[k].bbox,
                        ordered_list[k].aspect_ratio,
                        ordered_list[k].solidity,
                        ordered_list[k].score,
                    )
                    for k in range(4)
                )  # type: ignore[assignment]
                corners_px = oriented_pts
                break

    if base_color_image is not None and base_color_image.ndim == 3:
        debug = base_color_image.copy()
    elif gray_image is not None:
        debug = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    else:
        debug = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

    for c in candidates:
        x, y, bw, bh = c.bbox
        cv2.rectangle(debug, (x, y), (x + bw, y + bh), (220, 180, 0), 1)

    poly_pts = corners_px.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(debug, [poly_pts], isClosed=True, color=(0, 200, 0), thickness=3, lineType=cv2.LINE_AA)

    labels_short = ("C1", "C2", "C3", "C4")
    for idx, m in enumerate(selected):
        x, y, bw, bh = m.bbox
        cx_i, cy_i = int(round(m.cx)), int(round(m.cy))
        cv2.rectangle(debug, (x, y), (x + bw, y + bh), (0, 230, 0), 3)
        cv2.circle(debug, (cx_i, cy_i), 6, (0, 0, 255), -1)
        cv2.putText(
            debug,
            f"{labels_short[idx]} ({cx_i},{cy_i})",
            (max(10, x - 10), max(25, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 140, 0),
            2,
            cv2.LINE_AA,
        )

    return MarkerDetectionResult(
        markers=selected,
        corners_px=corners_px,
        candidates_count=len(candidates),
        morph_image=morph,
        debug_image=debug,
    )
