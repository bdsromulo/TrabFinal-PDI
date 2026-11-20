"""End-to-end OMR Pipeline orchestrator collecting all 9 stages and intermediate visual artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .bubbles import SheetReading, read_sheet_bubbles
from .grade import (
    ExamConfig,
    GradingResult,
    create_default_exam_config,
    grade_sheet,
    render_graded_sheet,
)
from .markers import MarkerDetectionResult, detect_corner_markers
from .preprocess import BinarizeMethod, DenoiseMethod, PreprocessConfig, PreprocessResult, preprocess_image
from .rectify import RectificationResult, rectify_sheet
from .sheet import SheetLayout, get_default_layout


@dataclass(frozen=True)
class PipelineConfig:
    """Complete configuration for the 9-stage OMR pipeline."""

    denoise_method: DenoiseMethod = "gaussian"
    gaussian_ksize: int = 5
    median_ksize: int = 5
    normalize_illumination: bool = True
    binarize_method: BinarizeMethod = "adaptive"
    adaptive_block_size: int = 75
    adaptive_c: int = 18
    inner_radius_ratio: float = 0.74
    min_fill_threshold: float = 0.22
    fixed_fill_threshold: Optional[float] = None
    fallback_exam_type: Optional[str] = "A"


@dataclass
class PipelineArtifacts:
    """All intermediate images and data structures generated during OMR execution."""

    original_bgr: np.ndarray
    preprocess: PreprocessResult
    markers: MarkerDetectionResult
    rectification: RectificationResult
    reading: SheetReading
    grading: GradingResult
    graded_image: np.ndarray


class OMRPipeline:
    """Main OMR pipeline executing stages 1 through 9."""

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        layout: Optional[SheetLayout] = None,
        exam_config: Optional[ExamConfig] = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.layout = layout or get_default_layout()
        self.exam_config = exam_config or create_default_exam_config()

    def process_image(self, image: np.ndarray) -> Tuple[GradingResult, PipelineArtifacts]:
        """Processes a single BGR or grayscale image array through all 9 OMR stages."""
        if image is None or image.size == 0:
            raise ValueError("Imagem de entrada invalida ou vazia.")

        if image.ndim == 2:
            original_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 4:
            original_bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        else:
            original_bgr = image.copy()

        # Stages 2 & 3: Preprocessing and Binarization
        pre_cfg = PreprocessConfig(
            denoise_method=self.config.denoise_method,
            gaussian_ksize=self.config.gaussian_ksize,
            median_ksize=self.config.median_ksize,
            normalize_illumination=self.config.normalize_illumination,
            binarize_method=self.config.binarize_method,
            adaptive_block_size=self.config.adaptive_block_size,
            adaptive_c=self.config.adaptive_c,
        )
        pre_res = preprocess_image(original_bgr, pre_cfg)

        # Stage 4: Fiducial Corner Marker Detection
        marker_res = detect_corner_markers(
            binary_image=pre_res.binary,
            gray_image=pre_res.denoised,
            base_color_image=original_bgr,
        )

        # Stage 5: Projective Rectification (Homography)
        rect_res = rectify_sheet(
            image=original_bgr,
            src_corners_px=marker_res.corners_px,
            layout=self.layout,
            binarize_method=self.config.binarize_method,
            normalize_illumination=self.config.normalize_illumination,
            adaptive_block_size=self.config.adaptive_block_size,
            adaptive_c=self.config.adaptive_c,
        )

        # Stages 6, 7 & 8: Bubble ROI Sampling, Mark Decision, and Exam Type Reading
        reading = read_sheet_bubbles(
            rectified_binary=rect_res.rectified_binary,
            rectified_gray=rect_res.rectified_normalized,
            layout=self.layout,
            inner_radius_ratio=self.config.inner_radius_ratio,
            min_fill_threshold=self.config.min_fill_threshold,
            fixed_threshold=self.config.fixed_fill_threshold,
        )

        # Stage 9: Grading & Visual Overlay
        grading = grade_sheet(
            reading=reading,
            exam_config=self.exam_config,
            fallback_exam_type=self.config.fallback_exam_type,
        )
        graded_img = render_graded_sheet(
            rectified_bgr=rect_res.rectified_color,
            reading=reading,
            grading=grading,
            layout=self.layout,
        )

        artifacts = PipelineArtifacts(
            original_bgr=original_bgr,
            preprocess=pre_res,
            markers=marker_res,
            rectification=rect_res,
            reading=reading,
            grading=grading,
            graded_image=graded_img,
        )
        return grading, artifacts

    def process_file(self, image_path: str | Path) -> Tuple[GradingResult, PipelineArtifacts]:
        """Reads an image file from disk (supporting Unicode paths on Windows) and processes it."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Arquivo de imagem nao encontrado: {path}")

        raw_bytes = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Nao foi possivel decodificar a imagem: {path}")

        return self.process_image(image)
