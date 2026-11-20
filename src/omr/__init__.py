"""OMR Exam Grader — Optical Mark Recognition for multiple-choice answer sheets.

Course project for Digital Image Processing (PDI — IF69D / OPEET014), UTFPR, 2026-2.
Authors: Rômulo Barbosa da Silva and Guilherme Oliver.
"""

from .sheet import BubbleCoord, MarkerCoord, SheetLayout, create_custom_layout, get_default_layout
from .preprocess import PreprocessConfig, PreprocessResult, preprocess_image, binarize_image
from .markers import DetectedMarker, MarkerDetectionError, MarkerDetectionResult, detect_corner_markers
from .rectify import RectificationError, RectificationResult, rectify_sheet
from .bubbles import BubbleReading, QuestionReading, SheetReading, read_sheet_bubbles
from .grade import AnswerKey, ExamConfig, QuestionGrade, GradingResult, grade_sheet, render_graded_sheet
from .pipeline import OMRPipeline, PipelineConfig, PipelineArtifacts
from .analytics import ClassAnalyticsReport, QuestionItemAnalysis, compute_class_analytics

__all__ = [
    "BubbleCoord",
    "MarkerCoord",
    "SheetLayout",
    "get_default_layout",
    "create_custom_layout",
    "PreprocessConfig",
    "PreprocessResult",
    "preprocess_image",
    "binarize_image",
    "DetectedMarker",
    "MarkerDetectionError",
    "MarkerDetectionResult",
    "detect_corner_markers",
    "RectificationError",
    "RectificationResult",
    "rectify_sheet",
    "BubbleReading",
    "QuestionReading",
    "SheetReading",
    "read_sheet_bubbles",
    "AnswerKey",
    "ExamConfig",
    "QuestionGrade",
    "GradingResult",
    "grade_sheet",
    "render_graded_sheet",
    "OMRPipeline",
    "PipelineConfig",
    "PipelineArtifacts",
    "ClassAnalyticsReport",
    "QuestionItemAnalysis",
    "compute_class_analytics",
]
