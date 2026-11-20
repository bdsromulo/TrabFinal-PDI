"""Classroom Psychometric & Classical Test Theory (CTT) Item Analysis for OMR Batch Grading.

Extends the OMR Exam Grader beyond individual sheet scoring to full class-level assessment:
- Score summary statistics (mean, std, median, min, max, quartiles, pass rate)
- Per-question Difficulty Index `p_i` (proportion of correct responses)
- Per-question Discrimination Index `D_i` (top 27% vs. bottom 27% performance gap)
- Per-question Distractor distribution (`A`..`E`, `BLANK`, `MULTIPLE`)
- Internal consistency reliability via Cronbach's Alpha (`alpha`)
- Automated quality alerts (e.g., possible answer key error when `D_i < 0` and a distractor dominates)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence

import numpy as np

from .grade import GradingResult


@dataclass(frozen=True)
class QuestionItemAnalysis:
    """Classical Test Theory (CTT) psychometric metrics for a single exam question."""

    question_number: int
    expected_option: str
    difficulty_index: float  # p_i in [0, 1] (higher = easier)
    discrimination_index: float  # D_i in [-1, 1]
    difficulty_category: str  # "Fácil", "Média", "Difícil", "Muito Difícil"
    option_frequencies: Dict[str, float]  # "A".."E", "BLANK", "MULTIPLE" -> proportion in [0, 1]
    most_frequent_marked: str
    alert: str  # "" or pedagogical/key-review warning


@dataclass
class ClassAnalyticsReport:
    """Aggregated psychometric and statistical report for a batch of graded answer sheets."""

    num_students: int
    mean_score: float
    std_score: float
    median_score: float
    min_score: float
    max_score: float
    pass_rate_pct: float  # percentage of students with normalized score >= 6.0 / 10
    cronbach_alpha: float
    items: Dict[int, QuestionItemAnalysis]
    flagged_questions: List[int]

    def to_dict(self) -> Dict[str, object]:
        return {
            "num_students": self.num_students,
            "mean_score": round(self.mean_score, 3),
            "std_score": round(self.std_score, 3),
            "median_score": round(self.median_score, 3),
            "min_score": round(self.min_score, 3),
            "max_score": round(self.max_score, 3),
            "pass_rate_pct": round(self.pass_rate_pct, 2),
            "cronbach_alpha": round(self.cronbach_alpha, 4),
            "flagged_questions": self.flagged_questions,
            "items": {str(q): asdict(item) for q, item in sorted(self.items.items())},
        }


def _classify_difficulty(p: float) -> str:
    if p >= 0.75:
        return "Fácil"
    if p >= 0.45:
        return "Média"
    if p >= 0.25:
        return "Difícil"
    return "Muito Difícil"


def compute_class_analytics(
    results: Sequence[GradingResult],
    passing_grade_10: float = 6.0,
) -> ClassAnalyticsReport:
    """Computes Classical Test Theory (CTT) item analysis and class statistics across a batch of sheets."""
    n = len(results)
    if n == 0:
        raise ValueError("A lista de resultados para analise de turma esta vazia.")

    scores = np.array([r.total_score for r in results], dtype=np.float64)
    scores_10 = np.array([r.score_normalized_10 for r in results], dtype=np.float64)

    q_nums = sorted(results[0].questions.keys())
    k_items = len(q_nums)

    # Binary hit matrix X of shape (n_students, k_items) and points matrix P
    hit_matrix = np.zeros((n, k_items), dtype=np.float64)
    pts_matrix = np.zeros((n, k_items), dtype=np.float64)
    for s_idx, res in enumerate(results):
        for q_idx, q_num in enumerate(q_nums):
            qg = res.questions[q_num]
            hit_matrix[s_idx, q_idx] = 1.0 if qg.outcome == "CORRECT" else 0.0
            pts_matrix[s_idx, q_idx] = qg.earned_points

    # Cronbach's Alpha for internal consistency
    if n > 1 and k_items > 1:
        item_vars = np.var(pts_matrix, axis=0, ddof=1)
        total_var = float(np.var(np.sum(pts_matrix, axis=1), ddof=1))
        if total_var > 1e-9:
            cronbach_alpha = float((k_items / (k_items - 1.0)) * (1.0 - float(np.sum(item_vars)) / total_var))
        else:
            cronbach_alpha = 0.0
    else:
        cronbach_alpha = 0.0

    # Sort students by total score for upper/lower 27% discrimination index D_i
    order = np.argsort(scores)
    group_size = max(1, int(round(0.27 * n)))
    bottom_indices = order[:group_size]
    top_indices = order[-group_size:]

    items_report: Dict[int, QuestionItemAnalysis] = {}
    flagged: List[int] = []

    for q_idx, q_num in enumerate(q_nums):
        p_i = float(np.mean(hit_matrix[:, q_idx]))
        p_top = float(np.mean(hit_matrix[top_indices, q_idx]))
        p_bot = float(np.mean(hit_matrix[bottom_indices, q_idx]))
        d_i = p_top - p_bot

        # Option frequencies across all students
        counts: Dict[str, int] = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "BLANK": 0, "MULTIPLE": 0}
        expected_opt = results[0].questions[q_num].expected_option or "?"
        for res in results:
            qg = res.questions[q_num]
            if qg.outcome == "BLANK":
                counts["BLANK"] += 1
            elif qg.outcome == "MULTIPLE":
                counts["MULTIPLE"] += 1
            elif qg.marked_option in counts:
                counts[qg.marked_option] += 1

        freqs = {k: round(v / float(n), 4) for k, v in counts.items()}
        most_freq = max(("A", "B", "C", "D", "E"), key=lambda o: counts.get(o, 0))

        alert = ""
        if d_i < -0.05 and most_freq != expected_opt and freqs.get(most_freq, 0.0) > p_i:
            alert = (
                f"Discriminacao negativa (D={d_i:.2f}) com alternativa '{most_freq}' "
                f"mais marcada ({freqs[most_freq]*100:.0f}%) que o gabarito '{expected_opt}': "
                f"revisar possivel erro no gabarito."
            )
            flagged.append(q_num)
        elif p_i < 0.15:
            alert = f"Indice de acerto muito baixo ({p_i*100:.0f}%): questao muito dificil ou ambigua."
            flagged.append(q_num)

        items_report[q_num] = QuestionItemAnalysis(
            question_number=q_num,
            expected_option=expected_opt,
            difficulty_index=round(p_i, 4),
            discrimination_index=round(d_i, 4),
            difficulty_category=_classify_difficulty(p_i),
            option_frequencies=freqs,
            most_frequent_marked=most_freq,
            alert=alert,
        )

    pass_rate = float(np.mean(scores_10 >= passing_grade_10) * 100.0)

    return ClassAnalyticsReport(
        num_students=n,
        mean_score=float(np.mean(scores)),
        std_score=float(np.std(scores)),
        median_score=float(np.median(scores)),
        min_score=float(np.min(scores)),
        max_score=float(np.max(scores)),
        pass_rate_pct=pass_rate,
        cronbach_alpha=cronbach_alpha,
        items=items_report,
        flagged_questions=flagged,
    )
