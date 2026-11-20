"""Stage 9: Answer key management, per-question weighted grading, and visual diagnostic rendering.

Supports:
- Up to 4 exam types (A, B, C, D), each with its own 50-question answer key.
- Individual per-question point weights (e.g., uniform 0.2 pts each = 10.0 total, or custom weights).
- Full breakdown of correct, wrong, blank, and multiple-mark (invalidated) questions.
- Annotated visual overlay on the rectified sheet highlighting every bubble and summarizing the score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Dict, List, Literal, Mapping, Optional, Tuple

import cv2
import numpy as np

from .bubbles import QuestionReading, SheetReading
from .sheet import EXAM_TYPES, OPTIONS, SheetLayout, get_default_layout

OutcomeType = Literal["CORRECT", "WRONG", "BLANK", "MULTIPLE"]


@dataclass
class AnswerKey:
    """Answer key for a single exam type (1..50 -> 'A'..'E')."""

    exam_type: str
    answers: Dict[int, str]

    def __post_init__(self) -> None:
        self.exam_type = self.exam_type.upper().strip()
        normalized: Dict[int, str] = {}
        for k, v in self.answers.items():
            q_num = int(k)
            opt = str(v).upper().strip()
            if opt not in OPTIONS:
                raise ValueError(f"Alternativa invalida '{opt}' na questao {q_num} (esperado A..E).")
            normalized[q_num] = opt
        self.answers = normalized


@dataclass
class ExamConfig:
    """Multi-type answer keys (Types A..D) and per-question weights (1..50)."""

    keys: Dict[str, AnswerKey] = field(default_factory=dict)
    weights: Dict[int, float] = field(default_factory=dict)
    num_questions: int = 50

    def __post_init__(self) -> None:
        if not self.weights:
            self.weights = {q: 0.2 for q in range(1, self.num_questions + 1)}
        else:
            self.weights = {int(k): float(v) for k, v in self.weights.items()}

    @property
    def max_possible_score(self) -> float:
        return float(sum(self.weights.get(q, 0.0) for q in range(1, self.num_questions + 1)))

    def to_dict(self) -> Dict[str, object]:
        return {
            "num_questions": self.num_questions,
            "weights": {str(q): round(w, 4) for q, w in sorted(self.weights.items())},
            "keys": {
                t: {str(q): ans for q, ans in sorted(ak.answers.items())}
                for t, ak in sorted(self.keys.items())
            },
        }

    def save_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ExamConfig":
        num_q = int(data.get("num_questions", 50))
        raw_weights = data.get("weights", {})
        weights = (
            {int(k): float(v) for k, v in raw_weights.items()}
            if isinstance(raw_weights, Mapping)
            else {q: 0.2 for q in range(1, num_q + 1)}
        )
        raw_keys = data.get("keys", {})
        keys: Dict[str, AnswerKey] = {}
        if isinstance(raw_keys, Mapping):
            for t, ans_map in raw_keys.items():
                if isinstance(ans_map, Mapping):
                    keys[str(t).upper()] = AnswerKey(
                        exam_type=str(t).upper(),
                        answers={int(q): str(a) for q, a in ans_map.items()},
                    )
        return cls(keys=keys, weights=weights, num_questions=num_q)

    @classmethod
    def load_json(cls, path: str | Path) -> "ExamConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)


def create_default_exam_config(seed: int = 2026) -> ExamConfig:
    """Generates a deterministic 4-type (A..D), 50-question exam configuration for testing and demo."""
    rng = np.random.default_rng(seed)
    keys: Dict[str, AnswerKey] = {}
    for t in EXAM_TYPES:
        answers = {
            q: str(rng.choice(OPTIONS))
            for q in range(1, 51)
        }
        keys[t] = AnswerKey(exam_type=t, answers=answers)

    weights = {q: 0.2 for q in range(1, 51)}  # 50 * 0.2 = 10.0 points total
    return ExamConfig(keys=keys, weights=weights, num_questions=50)


@dataclass(frozen=True)
class QuestionGrade:
    """Detailed grading result for a single question."""

    question_number: int
    expected_option: Optional[str]
    marked_option: Optional[str]
    marked_options: Tuple[str, ...]
    outcome: OutcomeType
    weight: float
    earned_points: float
    fill_ratios: Dict[str, float]


@dataclass
class GradingResult:
    """Complete grading report for an answer sheet."""

    detected_exam_type: Optional[str]
    exam_type_status: str
    used_exam_type: Optional[str]
    total_score: float
    max_score: float
    score_normalized_10: float
    num_correct: int
    num_wrong: int
    num_blank: int
    num_multiple: int
    calibrated_threshold: float
    questions: Dict[int, QuestionGrade]

    def to_dict(self) -> Dict[str, object]:
        return {
            "detected_exam_type": self.detected_exam_type,
            "exam_type_status": self.exam_type_status,
            "used_exam_type": self.used_exam_type,
            "total_score": round(self.total_score, 4),
            "max_score": round(self.max_score, 4),
            "score_normalized_10": round(self.score_normalized_10, 2),
            "num_correct": self.num_correct,
            "num_wrong": self.num_wrong,
            "num_blank": self.num_blank,
            "num_multiple": self.num_multiple,
            "calibrated_threshold": round(self.calibrated_threshold, 4),
            "questions": {
                str(q): asdict(qg) for q, qg in sorted(self.questions.items())
            },
        }


def grade_sheet(
    reading: SheetReading,
    exam_config: ExamConfig,
    fallback_exam_type: Optional[str] = "A",
) -> GradingResult:
    """Grades a `SheetReading` against the `ExamConfig` answer keys and per-question weights."""
    detected_type = reading.exam_type.marked_option
    type_status = reading.exam_type.status

    used_type = detected_type if (detected_type and detected_type in exam_config.keys) else fallback_exam_type
    answer_key = exam_config.keys.get(used_type) if used_type else None

    question_grades: Dict[int, QuestionGrade] = {}
    total_score = 0.0
    max_score = 0.0
    n_correct = 0
    n_wrong = 0
    n_blank = 0
    n_multiple = 0

    for q_num in range(1, exam_config.num_questions + 1):
        q_read: QuestionReading = reading.questions[q_num]
        weight = float(exam_config.weights.get(q_num, 0.2))
        max_score += weight
        expected = answer_key.answers.get(q_num) if answer_key else None

        if q_read.status == "BLANK":
            outcome: OutcomeType = "BLANK"
            earned = 0.0
            n_blank += 1
        elif q_read.status == "MULTIPLE":
            outcome = "MULTIPLE"
            earned = 0.0
            n_multiple += 1
        else:
            if expected is not None and q_read.marked_option == expected:
                outcome = "CORRECT"
                earned = weight
                n_correct += 1
            else:
                outcome = "WRONG"
                earned = 0.0
                n_wrong += 1

        total_score += earned
        question_grades[q_num] = QuestionGrade(
            question_number=q_num,
            expected_option=expected,
            marked_option=q_read.marked_option,
            marked_options=q_read.marked_options,
            outcome=outcome,
            weight=weight,
            earned_points=earned,
            fill_ratios=q_read.fill_ratios,
        )

    score_10 = (10.0 * total_score / max_score) if max_score > 0 else 0.0

    return GradingResult(
        detected_exam_type=detected_type,
        exam_type_status=type_status,
        used_exam_type=used_type,
        total_score=total_score,
        max_score=max_score,
        score_normalized_10=score_10,
        num_correct=n_correct,
        num_wrong=n_wrong,
        num_blank=n_blank,
        num_multiple=n_multiple,
        calibrated_threshold=reading.calibrated_threshold,
        questions=question_grades,
    )


def render_graded_sheet(
    rectified_bgr: np.ndarray,
    reading: SheetReading,
    grading: GradingResult,
    layout: Optional[SheetLayout] = None,
) -> np.ndarray:
    """Renders a visual grading overlay on the rectified sheet.

    Color legend (BGR):
    - Green (0, 185, 0): Correct marked bubble
    - Red (0, 0, 225): Wrong marked bubble (with dashed/thin blue ring on the expected answer)
    - Orange (0, 145, 255): Multiple marks (invalidated)
    - Yellow-Gold (0, 190, 230): Blank question (expected answer highlighted)
    """
    if layout is None:
        layout = get_default_layout()

    canvas = rectified_bgr.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    # Highlight exam type reading
    for b in reading.exam_type.bubbles:
        if b.is_marked:
            color = (0, 185, 0) if reading.exam_type.status == "MARKED" else (0, 140, 255)
            cv2.circle(canvas, (b.cx_px, b.cy_px), b.radius_px + 3, color, 3, cv2.LINE_AA)

    # Highlight all 50 questions
    for q_num, qg in grading.questions.items():
        q_read = reading.questions[q_num]
        for b in q_read.bubbles:
            if qg.outcome == "CORRECT" and b.is_marked:
                cv2.circle(canvas, (b.cx_px, b.cy_px), b.radius_px + 2, (0, 185, 0), 3, cv2.LINE_AA)
            elif qg.outcome == "WRONG":
                if b.is_marked:
                    cv2.circle(canvas, (b.cx_px, b.cy_px), b.radius_px + 2, (0, 0, 230), 3, cv2.LINE_AA)
                elif b.option == qg.expected_option:
                    cv2.circle(canvas, (b.cx_px, b.cy_px), b.radius_px + 2, (210, 90, 0), 2, cv2.LINE_AA)
            elif qg.outcome == "MULTIPLE":
                if b.is_marked:
                    cv2.circle(canvas, (b.cx_px, b.cy_px), b.radius_px + 2, (0, 140, 255), 3, cv2.LINE_AA)
                elif b.option == qg.expected_option:
                    cv2.circle(canvas, (b.cx_px, b.cy_px), b.radius_px + 1, (210, 90, 0), 2, cv2.LINE_AA)
            elif qg.outcome == "BLANK":
                if b.option == qg.expected_option:
                    cv2.circle(canvas, (b.cx_px, b.cy_px), b.radius_px + 2, (0, 185, 240), 2, cv2.LINE_AA)

    # Draw summary banner at the bottom of the rectified sheet
    px = layout.px_per_mm
    banner_x0, banner_y0 = int(32 * px), int(272 * px)
    banner_x1, banner_y1 = int(178 * px), int(287 * px)
    cv2.rectangle(canvas, (banner_x0, banner_y0), (banner_x1, banner_y1), (245, 248, 250), -1)
    cv2.rectangle(canvas, (banner_x0, banner_y0), (banner_x1, banner_y1), (40, 80, 120), 2)

    type_label = grading.detected_exam_type or f"{grading.used_exam_type}* ({grading.exam_type_status})"
    summary_text = (
        f"TIPO: {type_label}  |  NOTA: {grading.total_score:.2f} / {grading.max_score:.2f} "
        f"({grading.score_normalized_10:.1f}/10)  |  "
        f"Acertos: {grading.num_correct}  Erros: {grading.num_wrong}  "
        f"Brancos: {grading.num_blank}  Duplas: {grading.num_multiple}"
    )
    cv2.putText(
        canvas,
        summary_text,
        (banner_x0 + int(3 * px), banner_y0 + int(9.5 * px)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54 * (px / 8.0),
        (20, 40, 70),
        max(1, int(round(1.8 * px / 8.0))),
        cv2.LINE_AA,
    )

    return canvas
