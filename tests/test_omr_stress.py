"""Stress-test suite pushing the OMR Exam Grader across extreme geometric, photometric, and usage cases.

Covers:
1. 360-degree orientation invariance (90 deg landscape, 180 deg upside-down, 270 deg, and 45 deg diagonal).
2. Colored ballpoint pens (Blue, Red, Green) and light HB pencil under warm indoor lighting tint.
3. Sloppy / non-standard student markings ('X' crosses, checkmarks, partial off-center fills).
4. Erasure smudges vs. genuine double marks (disambiguating light pencil residue from two dark marks).
5. Non-linear radial lens barrel distortion & slight paper curvature (compensated by micro-registration).
6. Parameterized / custom exam layouts (e.g., 20 questions x 4 options in 1 column; 80 questions x 5 options in 4 columns).
7. Class-level Psychometric & Classical Test Theory (CTT) Item Analysis (`compute_class_analytics`).
"""

from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))
if str(ROOT_DIR / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "tools"))

from omr.analytics import compute_class_analytics
from omr.grade import AnswerKey, ExamConfig, create_default_exam_config
from omr.pipeline import OMRPipeline, PipelineConfig
from omr.sheet import OPTIONS, create_custom_layout, get_default_layout, render_sheet_image
from synthetic_dataset import SyntheticTransformParams, create_synthetic_sample


@pytest.mark.parametrize("angle_deg", [90.0, 180.0, 270.0, -90.0, 45.0, -135.0])
def test_360_degree_arbitrary_rotation_invariance(angle_deg: float) -> None:
    """Verifies 100% grading accuracy even when the photo is taken sideways (90/270) or upside-down (180)."""
    layout = get_default_layout()
    exam_cfg = create_default_exam_config(seed=700)
    params = SyntheticTransformParams(
        rotation_deg=angle_deg,
        scale_factor=0.80,
        perspective_jitter_px=18.0,
        shadow_strength=0.20,
        gaussian_noise_sigma=3.0,
    )
    sample = create_synthetic_sample(
        f"rot360_{angle_deg}",
        exam_config=exam_cfg,
        exam_type="B",
        params=params,
        layout=layout,
        seed=701,
    )

    pipeline = OMRPipeline(layout=layout, exam_config=exam_cfg)
    grading, artifacts = pipeline.process_image(sample.image_bgr)

    assert grading.detected_exam_type == "B"
    matches = sum(
        1
        for q_num, true_opts in sample.true_marks.items()
        if set(artifacts.reading.questions[q_num].marked_options) == set(true_opts)
    )
    assert matches == 50


@pytest.mark.parametrize(
    "pen_name,bgr_color",
    [
        ("blue_pen", (195, 45, 15)),
        ("red_pen", (25, 25, 200)),
        ("green_pen", (25, 155, 20)),
        ("light_pencil", (112, 112, 112)),
    ],
)
def test_colored_pens_and_light_pencil_under_warm_lighting(
    pen_name: str, bgr_color: tuple[int, int, int]
) -> None:
    """Verifies chromatic-boost grayscale conversion handles blue, red, green ink and light pencil."""
    layout = get_default_layout()
    exam_cfg = create_default_exam_config(seed=800)
    true_answers = dict(exam_cfg.keys["A"].answers)

    img = render_sheet_image(
        layout=layout,
        exam_type_marked="A",
        marked_answers=true_answers,
        mark_color_bgr=bgr_color,
        fill_Jitter_px=2,
    )
    # Apply warm yellowish tungsten indoor lighting tint (attenuating blue channel slightly)
    warm_tint = np.array([0.84, 0.95, 1.0], dtype=np.float32)
    img_warm = np.clip(img.astype(np.float32) * warm_tint, 0, 255).astype(np.uint8)

    pipeline = OMRPipeline(layout=layout, exam_config=exam_cfg)
    grading, _ = pipeline.process_image(img_warm)

    assert grading.detected_exam_type == "A", f"Falha no tipo com {pen_name}"
    assert grading.num_correct == 50, f"Falha nas questoes com {pen_name}: {grading.num_correct}/50"


@pytest.mark.parametrize("mark_style", ["cross", "check", "partial"])
def test_sloppy_student_mark_styles_cross_check_partial(mark_style: str) -> None:
    """Verifies that students marking bubbles with 'X' crosses, checkmarks, or partial fills are graded accurately."""
    layout = get_default_layout()
    exam_cfg = create_default_exam_config(seed=900)
    true_answers = dict(exam_cfg.keys["D"].answers)

    img = render_sheet_image(
        layout=layout,
        exam_type_marked="D",
        marked_answers=true_answers,
        mark_style=mark_style,
        fill_Jitter_px=3,
    )

    pipeline = OMRPipeline(layout=layout, exam_config=exam_cfg)
    grading, _ = pipeline.process_image(img)

    assert grading.detected_exam_type == "D"
    assert grading.num_correct == 50


def test_erasure_smudge_disambiguation_vs_true_double_mark() -> None:
    """Verifies that a faint erased pencil smudge next to a strong dark mark is not falsely annulled,
    whereas two genuine marks are strictly annulled as MULTIPLE."""
    layout = get_default_layout()
    exam_cfg = create_default_exam_config(seed=950)
    true_answers = dict(exam_cfg.keys["A"].answers)

    img = render_sheet_image(
        layout=layout,
        exam_type_marked="A",
        marked_answers=true_answers,
        mark_intensity=25,
    )

    # Simulate an erased pencil smudge on Question 7, option E (while true answer is not E)
    q7_bubbles = {b.option: b for b in layout.get_question_bubbles(7)}
    smudge_opt = "E" if true_answers[7] != "E" else "A"
    sb = q7_bubbles[smudge_opt]
    # Paint a small, faint smudge inside smudge_opt
    cv2.circle(img, (sb.cx_px, sb.cy_px), int(sb.radius_px * 0.42), (70, 70, 70), -1)

    # Simulate a genuine double mark on Question 12
    q12_bubbles = {b.option: b for b in layout.get_question_bubbles(12)}
    second_opt = "D" if true_answers[12] != "D" else "B"
    db = q12_bubbles[second_opt]
    cv2.circle(img, (db.cx_px, db.cy_px), int(db.radius_px * 0.84), (30, 30, 30), -1)

    pipeline = OMRPipeline(layout=layout, exam_config=exam_cfg)
    grading, _ = pipeline.process_image(img)

    # Question 7 should recover the primary mark and score CORRECT
    assert grading.questions[7].outcome == "CORRECT"
    assert grading.questions[7].marked_option == true_answers[7]
    # Question 12 must be strictly annulled as MULTIPLE
    assert grading.questions[12].outcome == "MULTIPLE"
    assert grading.num_correct == 49
    assert grading.num_multiple == 1


def test_radial_barrel_lens_distortion_and_paper_curvature() -> None:
    """Simulates mild smartphone wide-angle barrel distortion / paper bowing shifting central bubbles."""
    layout = get_default_layout()
    exam_cfg = create_default_exam_config(seed=1010)
    true_answers = dict(exam_cfg.keys["C"].answers)

    img = render_sheet_image(
        layout=layout,
        exam_type_marked="C",
        marked_answers=true_answers,
    )

    # Apply smooth sinusoidal paper-wave displacement up to 4.5 pixels in the middle of the sheet
    h, w = img.shape[:2]
    yy, xx = np.indices((h, w), dtype=np.float32)
    map_x = xx + 4.2 * np.sin(np.pi * yy / float(h)) * np.sin(np.pi * xx / float(w))
    map_y = yy + 4.2 * np.sin(np.pi * yy / float(h)) * np.sin(np.pi * xx / float(w))
    curved_img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderValue=(255, 255, 255))

    pipeline = OMRPipeline(layout=layout, exam_config=exam_cfg)
    grading, _ = pipeline.process_image(curved_img)

    assert grading.detected_exam_type == "C"
    assert grading.num_correct == 50


@pytest.mark.parametrize(
    "num_q,num_opts,num_cols",
    [
        (20, 4, 1),  # 20-question quiz (A-D) in 1 column
        (80, 5, 4),  # 80-question vestibular exam (A-E) in 4 columns
    ],
)
def test_parameterized_custom_sheet_layouts(num_q: int, num_opts: int, num_cols: int) -> None:
    """Verifies that custom layouts (e.g. 20x4 quiz or 80x5 exam) render, rectify, and grade at 100%."""
    layout = create_custom_layout(num_questions=num_q, num_options=num_opts, num_columns=num_cols)
    rng = np.random.default_rng(1100 + num_q)
    opts = OPTIONS[:num_opts]

    answers = {q: str(rng.choice(opts)) for q in range(1, num_q + 1)}
    exam_cfg = ExamConfig(
        keys={"A": AnswerKey("A", answers)},
        weights={q: 10.0 / num_q for q in range(1, num_q + 1)},
        num_questions=num_q,
    )

    img = render_sheet_image(layout=layout, exam_type_marked="A", marked_answers=answers)
    pipeline = OMRPipeline(layout=layout, exam_config=exam_cfg)
    grading, _ = pipeline.process_image(img)

    assert grading.detected_exam_type == "A"
    assert grading.num_correct == num_q
    assert pytest.approx(grading.total_score, abs=1e-4) == 10.0


def test_class_psychometric_item_analysis() -> None:
    """Verifies Classical Test Theory (CTT) class analytics, discrimination index, and wrong-key alerts."""
    layout = get_default_layout()
    exam_cfg = create_default_exam_config(seed=1200)
    pipeline = OMRPipeline(layout=layout, exam_config=exam_cfg)
    rng = np.random.default_rng(1200)

    # Simulate 16 students with varying proficiency levels [0.35 .. 0.95]
    # Intentionally invert Question 15 so high-scoring students pick 'E' while the key says 'A'
    exam_cfg.keys["A"].answers[15] = "A"
    batch_results = []

    for s_idx in range(16):
        skill = 0.30 + 0.65 * (s_idx / 15.0)
        marks: dict[int, list[str]] = {}
        for q in range(1, 51):
            correct_opt = exam_cfg.keys["A"].answers[q]
            if q == 15:
                # Strong students pick 'E', weak students guess 'A'
                marks[q] = ["E"] if skill > 0.55 else ["A"]
            else:
                if float(rng.random()) < skill:
                    marks[q] = [correct_opt]
                else:
                    wrong = [o for o in OPTIONS if o != correct_opt]
                    marks[q] = [str(rng.choice(wrong))]

        img = render_sheet_image(layout=layout, exam_type_marked="A", marked_answers=marks)
        grading, _ = pipeline.process_image(img)
        batch_results.append(grading)

    report = compute_class_analytics(batch_results)
    assert report.num_students == 16
    assert 0.0 <= report.mean_score <= 10.0
    assert report.cronbach_alpha > 0.70
    # Question 15 should have negative discrimination and be flagged for answer key review
    assert report.items[15].discrimination_index < 0.0
    assert 15 in report.flagged_questions
