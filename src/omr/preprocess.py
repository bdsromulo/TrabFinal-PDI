"""Stage 2 & 3: Image preprocessing, denoising, illumination normalization, and binarization.

Applies classical Digital Image Processing (PDI) operations taught in weeks 1, 3, 4, 5, and 9:
- Grayscale conversion (Week 1)
- Contrast & illumination normalization via morphological background estimation / division / CLAHE (Weeks 3 & 9)
- Spatial neighborhood filtering: Gaussian low-pass and Median filtering for salt-and-pepper noise (Weeks 4 & 5)
- Global automatic thresholding (Otsu) vs. local adaptive Gaussian thresholding (Week 9)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import cv2
import numpy as np

DenoiseMethod = Literal["gaussian", "median", "bilateral", "none"]
BinarizeMethod = Literal["adaptive", "otsu"]


@dataclass(frozen=True)
class PreprocessConfig:
    """Configuration parameters for image preprocessing and binarization."""

    denoise_method: DenoiseMethod = "gaussian"
    gaussian_ksize: int = 5
    median_ksize: int = 5
    normalize_illumination: bool = True
    background_kernel_fraction: float = 0.045
    use_clahe: bool = False
    binarize_method: BinarizeMethod = "adaptive"
    adaptive_block_size: int = 75
    adaptive_c: int = 18
    morph_clean_ksize: int = 3


@dataclass
class PreprocessResult:
    """Intermediate and final images produced by the preprocessing stage."""

    gray: np.ndarray
    denoised: np.ndarray
    normalized: np.ndarray
    binary: np.ndarray
    otsu_threshold_value: Optional[float] = None


def to_grayscale(image: np.ndarray, chromatic_boost: bool = True) -> np.ndarray:
    """Converts a BGR/BGRA or grayscale image to 2D uint8 grayscale.

    When `chromatic_boost=True`, blends standard ITU-R BT.601 luminance with the minimum
    RGB channel `min(R, G, B)`. For achromatic paper/black ink (R ~ G ~ B), this is identical
    to standard grayscale; for colored ballpoint pens (blue, red, green ink), the minimum
    RGB channel is strongly depressed, boosting ink-to-paper contrast.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or None.")
    if image.ndim == 2:
        return image.copy()
    if image.ndim == 3 and image.shape[2] == 4:
        bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim == 3 and image.shape[2] == 3:
        bgr = image
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    luma = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if not chromatic_boost:
        return luma

    min_ch = np.min(bgr, axis=2)
    blended = ((luma.astype(np.uint16) + min_ch.astype(np.uint16)) // 2).astype(np.uint8)
    return blended


def apply_denoising(
    gray: np.ndarray,
    method: DenoiseMethod = "gaussian",
    gaussian_ksize: int = 5,
    median_ksize: int = 5,
) -> np.ndarray:
    """Applies spatial neighborhood smoothing to attenuate sensor and salt-and-pepper noise."""
    if method == "none":
        return gray.copy()
    if method == "gaussian":
        k = gaussian_ksize if gaussian_ksize % 2 == 1 else gaussian_ksize + 1
        return cv2.GaussianBlur(gray, (k, k), 0)
    if method == "median":
        k = median_ksize if median_ksize % 2 == 1 else median_ksize + 1
        return cv2.medianBlur(gray, k)
    if method == "bilateral":
        return cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
    raise ValueError(f"Unknown denoise_method: {method}")


def normalize_illumination_background(
    gray: np.ndarray,
    kernel_fraction: float = 0.095,
    use_clahe: bool = False,
) -> np.ndarray:
    """Normalizes uneven lighting and shadows using multi-scale morphological background division.

    Estimates the slowly-varying bright paper background via morphological closing with a
    structuring element larger than the 112x112 fiducial markers (computed at 1/4 spatial scale
    for O(1) speed and smoothness), then divides the image by the background estimate to
    flatten illumination gradients while keeping solid fiducial markers intact.
    """
    h, w = gray.shape[:2]
    scale = 0.25
    small_w = max(32, int(round(w * scale)))
    small_h = max(32, int(round(h * scale)))
    small = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA)

    min_dim_small = min(small_h, small_w)
    ksize = max(15, int(round(min_dim_small * kernel_fraction)))
    if ksize % 2 == 0:
        ksize += 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    bg_small = cv2.morphologyEx(small, cv2.MORPH_CLOSE, kernel)
    bg_small = cv2.GaussianBlur(bg_small, (0, 0), sigmaX=max(2.0, ksize * 0.25))
    background = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_CUBIC)

    # Ensure background estimate is at least a safe floor to prevent amplifying dark borders
    background = np.maximum(background, 40)

    # Division normalization: (gray / background) * 255
    normalized = cv2.divide(gray, background, scale=255)

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        normalized = clahe.apply(normalized)

    return normalized


def binarize_image(
    gray: np.ndarray,
    method: BinarizeMethod = "adaptive",
    adaptive_block_size: int = 35,
    adaptive_c: int = 15,
    morph_clean_ksize: int = 0,
) -> tuple[np.ndarray, Optional[float]]:
    """Binarizes a grayscale image so that dark ink/marks become foreground (255) and paper becomes 0.

    Returns:
        Tuple `(binary_image, otsu_threshold_value)`.
    """
    otsu_val: Optional[float] = None
    if method == "otsu":
        otsu_t, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        otsu_val = float(otsu_t)
    elif method == "adaptive":
        block = adaptive_block_size if adaptive_block_size % 2 == 1 else adaptive_block_size + 1
        block = max(11, block)
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=block,
            C=adaptive_c,
        )
    else:
        raise ValueError(f"Unknown binarize_method: {method}")

    if morph_clean_ksize and morph_clean_ksize >= 2:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_clean_ksize, morph_clean_ksize)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return binary, otsu_val


def preprocess_image(
    image: np.ndarray,
    config: Optional[PreprocessConfig] = None,
) -> PreprocessResult:
    """Executes the full preprocessing pipeline: grayscale -> denoise -> illumination normalization -> binarize."""
    if config is None:
        config = PreprocessConfig()

    gray = to_grayscale(image)
    denoised = apply_denoising(
        gray,
        method=config.denoise_method,
        gaussian_ksize=config.gaussian_ksize,
        median_ksize=config.median_ksize,
    )

    if config.normalize_illumination:
        normalized = normalize_illumination_background(
            denoised,
            kernel_fraction=config.background_kernel_fraction,
            use_clahe=config.use_clahe,
        )
    else:
        normalized = denoised.copy()

    # For global Otsu, normalized or denoised can be used; when comparing Otsu vs Adaptive
    # under uneven illumination, Otsu on the un-flattened denoised image highlights Otsu's
    # classic global-threshold limitation, whereas Adaptive handles local gradients directly.
    target_for_bin = denoised if (config.binarize_method == "otsu" and not config.normalize_illumination) else normalized

    binary, otsu_val = binarize_image(
        target_for_bin,
        method=config.binarize_method,
        adaptive_block_size=config.adaptive_block_size,
        adaptive_c=config.adaptive_c,
        morph_clean_ksize=config.morph_clean_ksize,
    )

    return PreprocessResult(
        gray=gray,
        denoised=denoised,
        normalized=normalized,
        binary=binary,
        otsu_threshold_value=otsu_val,
    )
