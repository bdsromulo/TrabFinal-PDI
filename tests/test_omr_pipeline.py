"""Automated unit and integration tests for all 9 stages of the OMR Exam Grader pipeline."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))
if str(ROOT_DIR / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "tools"))

from omr.bubbles import read_sheet_bubbles
from omr.grade import AnswerKey, ExamConfig, create_default_exam_config, grade_sheet
from omr.markers import MarkerDetectionError, detect_corner_markers
from omr.pipeline import OMRPipeline, PipelineConfig
from omr.preprocess import PreprocessConfig, preprocess_image
from omr.rectify import rectify_sheet
from omr.sheet import EXAM_TYPES, OPTIONS, get_default_layout, render_sheet_image
from synthetic_dataset import SyntheticTransformParams, create_synthetic_sample


def test_canonical_sheet_layout_dimensions() -> None:
    layout = get_default_layout(px_per_mm=8.0)
    assert layout.canonical_width == 1680
    assert layout.canonical_height == 2376
    assert len(layout.markers) == 4
    assert len(layout.exam_type_bubbles) == 4
    assert len(layout.question_bubbles) == 250  # 50 questions x 5 options

    for q in range(1, 51):
        q_bubbles = layout.get_question_bubbles(q)
        assert len(q_bubbles) == 5
        assert tuple(b.option for b in q_bubbles) == OPTIONS


def test_clean_canonical_sheet_100_percent_accuracy() -> None:
    layout = get_default_layout()
    exam_cfg = create_default_exam_config(seed=101)
    true_answers = dict(exam_cfg.keys["C"].answers)

    img = render_sheet_image(
        layout=layout,
        exam_type_marked="C",
        marked_answers=true_answers,
    )

    pipeline = OMRPipeline(layout=layout, exam_config=exam_cfg)
    grading, _ = pipeline.process_image(img)

    assert grading.detected_exam_type == "C"
    assert grading.num_correct == 50
    assert grading.num_wrong == 0
    assert grading.num_blank == 0
    assert grading.num_multiple == 0
    assert pytest.approx(grading.total_score, abs=1e-5) == 10.0


def test_blank_and_multiple_marks_detection() -> None:
    layout = get_default_layout()
    exam_cfg = create_default_exam_config(seed=202)
    answers: dict[int, str | list[str]] = {
        q: exam_cfg.keys["A"].answers[q] for q in range(1, 51)
    }
    # Set 3 questions blank and 2 questions with double marks
    answers[5] = []
    answers[18] = []
    answers[42] = []
    answers[10] = ["A", "D"]
    answers[33] = ["B", "E"]

    img = render_sheet_image(
        layout=layout,
        exam_type_marked="A",
        marked_answers=answers,
    )

    pipeline = OMRPipeline(layout=layout, exam_config=exam_cfg)
    grading, _ = pipeline.process_image(img)

    assert grading.detected_exam_type == "A"
    assert grading.num_blank == 3
    assert grading.num_multiple == 2
    assert grading.num_correct == 45
    assert grading.questions[5].outcome == "BLANK"
    assert grading.questions[10].outcome == "MULTIPLE"
    assert set(grading.questions[10].marked_options) == {"A", "D"}


def test_all_blank_sheet_has_zero_false_positives() -> None:
    layout = get_default_layout()
    exam_cfg = create_default_exam_config(seed=303)
    blank_img = render_sheet_image(layout=layout)

    pipeline = OMRPipeline(layout=layout, exam_config=exam_cfg)
    grading, _ = pipeline.process_image(blank_img)

    assert grading.exam_type_status == "BLANK"
    assert grading.num_blank == 50
    assert grading.num_correct == 0
    assert grading.num_wrong == 0
    assert grading.num_multiple == 0
    assert grading.total_score == 0.0


def test_custom_per_question_weights() -> None:
    layout = get_default_layout()
    exam_cfg = create_default_exam_config(seed=404)
    # Give question 1 a weight of 5.0 and all other 49 questions a weight of 0.1
    exam_cfg.weights = {q: (5.0 if q == 1 else 0.1) for q in range(1, 51)}

    # Only mark question 1 correctly, leave 2..50 blank
    marks: dict[int, str | list[str]] = {1: exam_cfg.keys["B"].answers[1]}
    img = render_sheet_image(layout=layout, exam_type_marked="B", marked_answers=marks)

    pipeline = OMRPipeline(layout=layout, exam_config=exam_cfg)
    grading, _ = pipeline.process_image(img)

    assert grading.detected_exam_type == "B"
    assert grading.num_correct == 1
    assert grading.num_blank == 49
    assert pytest.approx(grading.total_score, abs=1e-5) == 5.0
    assert pytest.approx(grading.max_score, abs=1e-5) == 9.9


def test_perspective_shadow_and_noise_robustness() -> None:
    layout = get_default_layout()
    exam_cfg = create_default_exam_config(seed=505)
    params = SyntheticTransformParams(
        rotation_deg=-9.5,
        scale_factor=0.83,
        perspective_jitter_px=34.0,
        shadow_strength=0.45,
        shadow_angle_deg=40.0,
        gaussian_noise_sigma=6.0,
    )
    sample = create_synthetic_sample(
        "test_degraded",
        exam_config=exam_cfg,
        exam_type="D",
        params=params,
        layout=layout,
        seed=505,
    )

    pipeline = OMRPipeline(layout=layout, exam_config=exam_cfg)
    grading, artifacts = pipeline.process_image(sample.image_bgr)

    assert grading.detected_exam_type == "D"
    # Verify corner detection accuracy (< 6 px RMSE)
    rmse = float(
        np.sqrt(np.mean(np.sum((artifacts.markers.corners_px - sample.true_corners_px) ** 2, axis=1)))
    )
    assert rmse < 6.0

    # Verify all 50 questions match ground truth
    matches = sum(
        1
        for q_num, true_opts in sample.true_marks.items()
        if set(artifacts.reading.questions[q_num].marked_options) == set(true_opts)
    )
    assert matches == 50


def test_missing_markers_raises_clear_error() -> None:
    empty_white = np.full((1200, 900, 3), 255, dtype=np.uint8)
    pipeline = OMRPipeline()
    with pytest.raises(MarkerDetectionError):
        pipeline.process_image(empty_white)
