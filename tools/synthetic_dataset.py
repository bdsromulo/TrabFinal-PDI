"""Synthetic Smartphone Photo & Degraded Answer Sheet Generator for OMR Testing and Evaluation.

Simulates realistic smartphone captures of filled answer sheets:
- Random or specified student answers (including blank questions and double marks)
- Projective perspective warp (3D camera tilt / homography), in-plane rotation, and scaling
- Tabletop background surrounding the A4 paper
- Uneven illumination and directional shadows (linear/radial lighting gradients)
- Additive Gaussian sensor noise and impulsive salt-and-pepper noise

Usage:
    python tools/synthetic_dataset.py
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

from omr.grade import ExamConfig, create_default_exam_config
from omr.sheet import EXAM_TYPES, OPTIONS, SheetLayout, get_default_layout, render_sheet_image


@dataclass
class SyntheticTransformParams:
    """Parameters controlling geometric and photometric degradation of a synthetic sheet photo."""

    rotation_deg: float = 0.0
    scale_factor: float = 0.86
    perspective_jitter_px: float = 28.0
    shadow_strength: float = 0.25  # 0.0 (uniform) to 0.75 (heavy shadow)
    shadow_angle_deg: float = 35.0
    gaussian_noise_sigma: float = 4.0
    salt_pepper_prob: float = 0.0
    blur_ksize: int = 3
    mark_intensity: int = 32
    background_gray: int = 105


@dataclass
class SyntheticSheetSample:
    """A generated synthetic photo alongside its exact ground-truth labels."""

    sample_id: str
    image_bgr: np.ndarray
    true_exam_type: str
    true_marks: Dict[int, List[str]]  # 1..50 -> [] (blank), ["B"] (single), ["A", "D"] (multiple)
    true_corners_px: np.ndarray  # 4x2 float32 marker centers in the warped image
    params: SyntheticTransformParams


def generate_student_marks(
    exam_config: ExamConfig,
    exam_type: str = "A",
    target_accuracy: float = 0.78,
    blank_prob: float = 0.04,
    multiple_prob: float = 0.04,
    rng: Optional[np.random.Generator] = None,
) -> Dict[int, List[str]]:
    """Simulates a student's markings on the 50 questions with controlled accuracy, blanks, and double marks."""
    if rng is None:
        rng = np.random.default_rng(12345)

    key = exam_config.keys[exam_type].answers
    marks: Dict[int, List[str]] = {}

    for q_num in range(1, exam_config.num_questions + 1):
        u = float(rng.random())
        correct_opt = key[q_num]
        if u < blank_prob:
            marks[q_num] = []
        elif u < blank_prob + multiple_prob:
            other_opts = [o for o in OPTIONS if o != correct_opt]
            second_opt = str(rng.choice(other_opts))
            marks[q_num] = sorted([correct_opt, second_opt])
        elif u < blank_prob + multiple_prob + target_accuracy * (1.0 - blank_prob - multiple_prob):
            marks[q_num] = [correct_opt]
        else:
            wrong_opts = [o for o in OPTIONS if o != correct_opt]
            marks[q_num] = [str(rng.choice(wrong_opts))]

    return marks


def apply_photometric_and_geometric_degradation(
    canonical_sheet_bgr: np.ndarray,
    layout: SheetLayout,
    params: SyntheticTransformParams,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Applies perspective warp, rotation, scale, illumination gradient, and noise to a canonical sheet.

    Returns:
        Tuple `(degraded_bgr_image, transformed_marker_centers_4x2)`.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    h, w = canonical_sheet_bgr.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    # 1. Build composite similarity (rotation + scale) around center + projective corner jitter
    src_page_corners = np.array(
        [[0.0, 0.0], [float(w), 0.0], [float(w), float(h)], [0.0, float(h)]],
        dtype=np.float32,
    )

    theta = np.deg2rad(params.rotation_deg)
    cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))
    rot_mat = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32) * params.scale_factor

    centered = src_page_corners - np.array([[cx, cy]], dtype=np.float32)
    rotated = centered @ rot_mat.T + np.array([[cx, cy]], dtype=np.float32)

    if params.perspective_jitter_px > 0:
        jitter = rng.uniform(
            -params.perspective_jitter_px,
            params.perspective_jitter_px,
            size=(4, 2),
        ).astype(np.float32)
        dst_page_corners = rotated + jitter
    else:
        dst_page_corners = rotated

    # Ensure positive margin inside canvas; expand canvas if needed for large rotations
    min_xy = dst_page_corners.min(axis=0)
    max_xy = dst_page_corners.max(axis=0)
    pad = 50.0
    shift_x = max(0.0, pad - float(min_xy[0]))
    shift_y = max(0.0, pad - float(min_xy[1]))
    dst_page_corners[:, 0] += shift_x
    dst_page_corners[:, 1] += shift_y

    out_w = int(max(w, np.ceil(float(max_xy[0]) + shift_x + pad)))
    out_h = int(max(h, np.ceil(float(max_xy[1]) + shift_y + pad)))

    H_sim, _ = cv2.findHomography(src_page_corners, dst_page_corners)

    # Create a textured wooden/desk background so the sheet sits on a realistic surface
    bg = np.full((out_h, out_w, 3), params.background_gray, dtype=np.uint8)
    warped = cv2.warpPerspective(
        canonical_sheet_bgr,
        H_sim,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_TRANSPARENT,
        dst=bg,
    )

    # Transform the 4 canonical fiducial marker centers through H_sim
    canon_markers = layout.marker_centers_px.reshape(-1, 1, 2)
    true_markers = cv2.perspectiveTransform(canon_markers, H_sim).reshape(4, 2)

    img_f = warped.astype(np.float32)

    # 2. Apply directional illumination / shadow gradient
    if params.shadow_strength > 1e-4:
        yy, xx = np.mgrid[0:out_h, 0:out_w].astype(np.float32)
        ang = np.deg2rad(params.shadow_angle_deg)
        proj = (xx / out_w) * float(np.cos(ang)) + (yy / out_h) * float(np.sin(ang))
        proj_norm = (proj - proj.min()) / max(1e-6, float(proj.max() - proj.min()))
        # Non-linear shadow profile (simulating a smartphone/hand shadow cast across the sheet)
        illum_map = 1.0 - params.shadow_strength * (0.65 * proj_norm + 0.35 * (proj_norm ** 2))
        img_f = img_f * illum_map[:, :, None]

    # 3. Optional slight optical blur
    if params.blur_ksize and params.blur_ksize >= 3:
        k = params.blur_ksize if params.blur_ksize % 2 == 1 else params.blur_ksize + 1
        img_f = cv2.GaussianBlur(img_f, (k, k), 0)

    # 4. Additive Gaussian noise
    if params.gaussian_noise_sigma > 0:
        noise = rng.normal(0.0, params.gaussian_noise_sigma, size=img_f.shape).astype(np.float32)
        img_f = img_f + noise

    out_img = np.clip(img_f, 0, 255).astype(np.uint8)

    # 5. Impulsive salt-and-pepper noise
    if params.salt_pepper_prob > 0:
        sp_u = rng.random(size=(out_h, out_w))
        half_p = params.salt_pepper_prob / 2.0
        out_img[sp_u < half_p] = 0
        out_img[(sp_u >= half_p) & (sp_u < params.salt_pepper_prob)] = 255

    return out_img, true_markers


def create_synthetic_sample(
    sample_id: str,
    exam_config: ExamConfig,
    exam_type: str = "A",
    marks: Optional[Dict[int, List[str]]] = None,
    params: Optional[SyntheticTransformParams] = None,
    layout: Optional[SheetLayout] = None,
    seed: int = 42,
) -> SyntheticSheetSample:
    """Creates a complete synthetic smartphone photo sample with ground truth."""
    if layout is None:
        layout = get_default_layout()
    if params is None:
        params = SyntheticTransformParams()
    rng = np.random.default_rng(seed)

    if marks is None:
        marks = generate_student_marks(exam_config, exam_type=exam_type, rng=rng)

    canonical_img = render_sheet_image(
        layout=layout,
        exam_type_marked=exam_type,
        marked_answers=marks,
        mark_intensity=params.mark_intensity,
        fill_Jitter_px=2,
        rng=rng,
    )

    degraded_img, true_corners = apply_photometric_and_geometric_degradation(
        canonical_sheet_bgr=canonical_img,
        layout=layout,
        params=params,
        rng=rng,
    )

    return SyntheticSheetSample(
        sample_id=sample_id,
        image_bgr=degraded_img,
        true_exam_type=exam_type,
        true_marks=marks,
        true_corners_px=true_corners,
        params=params,
    )


def generate_benchmark_dataset(
    output_dir: str | Path,
    num_samples: int = 12,
    seed: int = 2026,
) -> List[Path]:
    """Generates a benchmark set of synthetic smartphone photos on disk with a ground-truth manifest."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    layout = get_default_layout()
    exam_cfg = create_default_exam_config(seed=seed)
    rng = np.random.default_rng(seed)

    manifest: List[dict] = []
    saved_paths: List[Path] = []

    scenarios = [
        ("01_ideal_tipoA", "A", SyntheticTransformParams(rotation_deg=0.5, scale_factor=0.92, perspective_jitter_px=10, shadow_strength=0.05, gaussian_noise_sigma=2.0)),
        ("02_perspectiva_tipoB", "B", SyntheticTransformParams(rotation_deg=-4.5, scale_factor=0.85, perspective_jitter_px=42, shadow_strength=0.18, gaussian_noise_sigma=4.0)),
        ("03_sombra_forte_tipoC", "C", SyntheticTransformParams(rotation_deg=3.0, scale_factor=0.88, perspective_jitter_px=25, shadow_strength=0.52, shadow_angle_deg=25, gaussian_noise_sigma=5.0)),
        ("04_rotacao_12deg_tipoD", "D", SyntheticTransformParams(rotation_deg=12.0, scale_factor=0.82, perspective_jitter_px=30, shadow_strength=0.22, gaussian_noise_sigma=4.0)),
        ("05_ruido_sal_pimenta_tipoA", "A", SyntheticTransformParams(rotation_deg=-2.0, scale_factor=0.88, perspective_jitter_px=20, shadow_strength=0.15, gaussian_noise_sigma=3.0, salt_pepper_prob=0.015)),
        ("06_ruido_gaussiano_tipoB", "B", SyntheticTransformParams(rotation_deg=6.0, scale_factor=0.84, perspective_jitter_px=28, shadow_strength=0.25, gaussian_noise_sigma=14.0)),
        ("07_marca_fraca_tipoC", "C", SyntheticTransformParams(rotation_deg=-7.5, scale_factor=0.86, perspective_jitter_px=32, shadow_strength=0.20, gaussian_noise_sigma=4.0, mark_intensity=85)),
        ("08_escala_reduzida_tipoD", "D", SyntheticTransformParams(rotation_deg=4.0, scale_factor=0.68, perspective_jitter_px=22, shadow_strength=0.20, gaussian_noise_sigma=3.5)),
    ]

    # Add extra randomized smartphone scenarios up to num_samples
    for i in range(len(scenarios) + 1, num_samples + 1):
        t = str(EXAM_TYPES[(i - 1) % len(EXAM_TYPES)])
        p = SyntheticTransformParams(
            rotation_deg=float(rng.uniform(-14.0, 14.0)),
            scale_factor=float(rng.uniform(0.74, 0.92)),
            perspective_jitter_px=float(rng.uniform(15.0, 45.0)),
            shadow_strength=float(rng.uniform(0.10, 0.48)),
            shadow_angle_deg=float(rng.uniform(0.0, 180.0)),
            gaussian_noise_sigma=float(rng.uniform(2.0, 10.0)),
            salt_pepper_prob=float(rng.choice([0.0, 0.0, 0.005])),
            mark_intensity=int(rng.integers(22, 75)),
        )
        scenarios.append((f"{i:02d}_smartphone_tipo{t}", t, p))

    for idx, (sample_id, exam_type, params) in enumerate(scenarios[:num_samples]):
        sample = create_synthetic_sample(
            sample_id=sample_id,
            exam_config=exam_cfg,
            exam_type=exam_type,
            params=params,
            layout=layout,
            seed=seed + idx * 97,
        )
        img_path = out_path / f"{sample_id}.png"
        ok, buf = cv2.imencode(".png", sample.image_bgr)
        if ok:
            buf.tofile(str(img_path))
        saved_paths.append(img_path)

        manifest.append(
            {
                "sample_id": sample.sample_id,
                "file": img_path.name,
                "true_exam_type": sample.true_exam_type,
                "true_marks": {str(q): opts for q, opts in sorted(sample.true_marks.items())},
                "true_corners_px": sample.true_corners_px.tolist(),
                "params": asdict(sample.params),
            }
        )

    manifest_path = out_path / "ground_truth.json"
    manifest_path.write_text(
        json.dumps({"samples": manifest}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return saved_paths


def main() -> int:
    out_dir = ROOT_DIR / "data" / "synthetic"
    paths = generate_benchmark_dataset(out_dir, num_samples=12, seed=2026)
    print(f"[OK] {len(paths)} fotos sinteticas + ground_truth.json gerados em: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
