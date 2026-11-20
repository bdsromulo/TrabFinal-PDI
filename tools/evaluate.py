"""Quantitative & Qualitative Evaluation Suite for the OMR Exam Grader (Weeks 5, 9 & 11).

Executes the full experimental evaluation planned in `docs/omr-visao-geral.typ` and exports
300 DPI publication-ready figures in Portuguese to `docs/relatorio/figuras/`:
1. `fig_pipeline_etapas.png`: Step-by-step qualitative visualization of all pipeline stages.
2. `fig_otsu_vs_adaptativo.png`: Quantitative & visual comparison of Global Otsu vs. Local Adaptive
   Thresholding under increasing illumination shadow gradients.
3. `fig_matriz_confusao.png`: Bubble-level and Question-level confusion matrices (A..E, Branco, Dupla).
4. `fig_curvas_robustez.png`: 4-panel robustness curves (Accuracy vs. Rotation Angle, Scale Factor,
   Gaussian Noise sigma, and Salt-and-Pepper noise probability, comparing Gaussian vs. Median filters).
5. `docs/relatorio/metricas_avaliacao.json`: Complete numerical metrics summary for the LaTeX report.

Usage:
    python tools/evaluate.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))
if str(ROOT_DIR / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "tools"))

from omr.grade import ExamConfig, create_default_exam_config
from omr.pipeline import OMRPipeline, PipelineConfig
from omr.sheet import OPTIONS, get_default_layout
from synthetic_dataset import (
    SyntheticSheetSample,
    SyntheticTransformParams,
    create_synthetic_sample,
)


def _question_label(opts: List[str] | Tuple[str, ...]) -> str:
    if len(opts) == 0:
        return "Branco"
    if len(opts) == 1:
        return opts[0]
    return "Dupla"


def evaluate_sample(
    sample: SyntheticSheetSample,
    pipeline: OMRPipeline,
) -> Dict[str, object]:
    """Runs the OMR pipeline on a synthetic sample and compares predictions against ground truth."""
    try:
        grading, artifacts = pipeline.process_image(sample.image_bgr)
    except Exception as exc:
        return {
            "marker_detected": False,
            "error": str(exc),
            "corner_rmse_px": None,
            "type_correct": False,
            "question_accuracy": 0.0,
            "bubble_accuracy": 0.0,
            "sheet_perfect": False,
            "bubble_pairs": [],
            "question_pairs": [],
        }

    # Corner localization RMSE in pixels
    pred_corners = artifacts.markers.corners_px
    true_corners = sample.true_corners_px
    corner_rmse = float(np.sqrt(np.mean(np.sum((pred_corners - true_corners) ** 2, axis=1))))

    type_correct = grading.detected_exam_type == sample.true_exam_type

    q_correct_count = 0
    b_correct_count = 0
    total_q = len(sample.true_marks)
    total_b = total_q * len(OPTIONS)

    bubble_pairs: List[Tuple[int, int]] = []  # (true_is_marked, pred_is_marked)
    question_pairs: List[Tuple[str, str]] = []  # (true_class, pred_class)

    for q_num, true_opts in sample.true_marks.items():
        q_read = artifacts.reading.questions[q_num]
        pred_opts = list(q_read.marked_options)

        true_cls = _question_label(true_opts)
        pred_cls = _question_label(pred_opts)
        question_pairs.append((true_cls, pred_cls))

        if set(true_opts) == set(pred_opts):
            q_correct_count += 1

        for opt in OPTIONS:
            t_mark = 1 if opt in true_opts else 0
            p_mark = 1 if opt in pred_opts else 0
            if t_mark == p_mark:
                b_correct_count += 1
            bubble_pairs.append((t_mark, p_mark))

    q_acc = q_correct_count / max(1, total_q)
    b_acc = b_correct_count / max(1, total_b)

    return {
        "marker_detected": True,
        "corner_rmse_px": corner_rmse,
        "type_correct": type_correct,
        "question_accuracy": q_acc,
        "bubble_accuracy": b_acc,
        "sheet_perfect": (q_correct_count == total_q and type_correct),
        "bubble_pairs": bubble_pairs,
        "question_pairs": question_pairs,
    }


def save_pipeline_stages_figure(
    exam_cfg: ExamConfig,
    fig_dir: Path,
) -> None:
    """Generates `fig_pipeline_etapas.png` showing the step-by-step images of the pipeline."""
    layout = get_default_layout()
    params = SyntheticTransformParams(
        rotation_deg=-6.5,
        scale_factor=0.85,
        perspective_jitter_px=38.0,
        shadow_strength=0.42,
        shadow_angle_deg=30.0,
        gaussian_noise_sigma=5.0,
    )
    sample = create_synthetic_sample(
        "demo_etapas",
        exam_config=exam_cfg,
        exam_type="B",
        params=params,
        layout=layout,
        seed=2026,
    )
    pipeline = OMRPipeline(config=PipelineConfig(), layout=layout, exam_config=exam_cfg)
    _, art = pipeline.process_image(sample.image_bgr)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9.2), dpi=300)
    stages = [
        (cv2.cvtColor(art.original_bgr, cv2.COLOR_BGR2RGB), "(a) Foto adquirida (perspectiva + sombra)"),
        (art.preprocess.normalized, "(b) Pré-processamento e normalização de fundo"),
        (art.preprocess.binary, "(c) Binarização adaptativa local"),
        (cv2.cvtColor(art.markers.debug_image, cv2.COLOR_BGR2RGB), "(d) Detecção dos 4 marcadores fiduciais"),
        (cv2.cvtColor(art.rectification.grid_overlay, cv2.COLOR_BGR2RGB), "(e) Retificação por homografia + grade ROI"),
        (cv2.cvtColor(art.graded_image, cv2.COLOR_BGR2RGB), "(f) Leitura das bolhas e correção final"),
    ]

    for ax, (img, title) in zip(axes.flat, stages):
        if img.ndim == 2:
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=10.5, fontweight="bold", pad=6)
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(fig_dir / "fig_pipeline_etapas.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_otsu_vs_adaptive_experiment(
    exam_cfg: ExamConfig,
    fig_dir: Path,
) -> Dict[str, object]:
    """Compares Global Otsu (without background division) vs. Local Adaptive + Background Normalization."""
    layout = get_default_layout()
    shadow_levels = [0.0, 0.15, 0.30, 0.45, 0.60, 0.72]
    trials_per_level = 4

    cfg_otsu_raw = PipelineConfig(
        binarize_method="otsu",
        normalize_illumination=False,
    )
    cfg_otsu_norm = PipelineConfig(
        binarize_method="otsu",
        normalize_illumination=True,
    )
    cfg_adaptive = PipelineConfig(
        binarize_method="adaptive",
        normalize_illumination=True,
    )

    acc_otsu_raw: List[float] = []
    acc_otsu_norm: List[float] = []
    acc_adaptive: List[float] = []

    for s_idx, shadow in enumerate(shadow_levels):
        q_raw, q_norm, q_adap = [], [], []
        for t in range(trials_per_level):
            params = SyntheticTransformParams(
                rotation_deg=2.5,
                scale_factor=0.88,
                perspective_jitter_px=18.0,
                shadow_strength=shadow,
                shadow_angle_deg=25.0 + 30.0 * t,
                gaussian_noise_sigma=4.0,
            )
            sample = create_synthetic_sample(
                f"shadow_{s_idx}_{t}",
                exam_config=exam_cfg,
                exam_type="A",
                params=params,
                layout=layout,
                seed=500 + s_idx * 17 + t,
            )
            r_raw = evaluate_sample(sample, OMRPipeline(cfg_otsu_raw, layout, exam_cfg))
            r_norm = evaluate_sample(sample, OMRPipeline(cfg_otsu_norm, layout, exam_cfg))
            r_adap = evaluate_sample(sample, OMRPipeline(cfg_adaptive, layout, exam_cfg))

            q_raw.append(float(r_raw["question_accuracy"]) * 100.0)
            q_norm.append(float(r_norm["question_accuracy"]) * 100.0)
            q_adap.append(float(r_adap["question_accuracy"]) * 100.0)

        acc_otsu_raw.append(float(np.mean(q_raw)))
        acc_otsu_norm.append(float(np.mean(q_norm)))
        acc_adaptive.append(float(np.mean(q_adap)))

    # Also generate side-by-side visual comparison under strong shadow (0.60)
    heavy_sample = create_synthetic_sample(
        "heavy_shadow",
        exam_config=exam_cfg,
        exam_type="A",
        params=SyntheticTransformParams(
            rotation_deg=0.0,
            scale_factor=0.92,
            perspective_jitter_px=5.0,
            shadow_strength=0.62,
            shadow_angle_deg=35.0,
            gaussian_noise_sigma=3.0,
        ),
        layout=layout,
        seed=999,
    )
    from omr.preprocess import PreprocessConfig, preprocess_image

    pre_otsu = preprocess_image(
        heavy_sample.image_bgr,
        PreprocessConfig(binarize_method="otsu", normalize_illumination=False),
    )
    pre_adap = preprocess_image(
        heavy_sample.image_bgr,
        PreprocessConfig(binarize_method="adaptive", normalize_illumination=True),
    )

    fig = plt.figure(figsize=(13.5, 4.6), dpi=300)
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 1.45])

    ax0 = fig.add_subplot(gs[0])
    ax0.imshow( pre_otsu.gray, cmap="gray", vmin=0, vmax=255)
    ax0.set_title("(a) Foto com gradiente de sombra", fontsize=10, fontweight="bold")
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[1])
    ax1.imshow(pre_otsu.binary, cmap="gray", vmin=0, vmax=255)
    ax1.set_title("(b) Otsu Global direto", fontsize=10, fontweight="bold")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[2])
    ax2.imshow(pre_adap.binary, cmap="gray", vmin=0, vmax=255)
    ax2.set_title("(c) Normalização + Limiar Adaptativo", fontsize=10, fontweight="bold")
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[3])
    x_pct = [s * 100 for s in shadow_levels]
    ax3.plot(x_pct, acc_otsu_raw, "o--", color="#c0392b", linewidth=2.2, label="Otsu Global (sem norm.)")
    ax3.plot(x_pct, acc_otsu_norm, "s-.", color="#e67e22", linewidth=2.0, label="Otsu + Norm. Morfológica")
    ax3.plot(x_pct, acc_adaptive, "^-", color="#1f4e79", linewidth=2.4, label="Limiar Adaptativo + Norm.")
    ax3.set_title("(d) Acurácia vs. Intensidade de Sombra", fontsize=10, fontweight="bold")
    ax3.set_xlabel("Intensidade do gradiente de sombra (%)", fontsize=9.5)
    ax3.set_ylabel("Acurácia por questão (%)", fontsize=9.5)
    ax3.set_ylim(0, 105)
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(fontsize=8.2, loc="lower left")

    plt.tight_layout()
    fig.savefig(fig_dir / "fig_otsu_vs_adaptativo.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "shadow_levels": shadow_levels,
        "acc_otsu_raw_pct": [round(v, 2) for v in acc_otsu_raw],
        "acc_otsu_norm_pct": [round(v, 2) for v in acc_otsu_norm],
        "acc_adaptive_pct": [round(v, 2) for v in acc_adaptive],
    }


def run_benchmark_and_confusion_matrix(
    exam_cfg: ExamConfig,
    fig_dir: Path,
    num_sheets: int = 30,
) -> Dict[str, object]:
    """Evaluates `num_sheets` diverse synthetic smartphone photos and builds confusion matrices."""
    layout = get_default_layout()
    pipeline = OMRPipeline(config=PipelineConfig(), layout=layout, exam_config=exam_cfg)
    rng = np.random.default_rng(202611)

    all_bubble_pairs: List[Tuple[int, int]] = []
    all_question_pairs: List[Tuple[str, str]] = []
    corner_rmses: List[float] = []
    q_accs: List[float] = []
    b_accs: List[float] = []
    markers_ok = 0
    types_ok = 0
    sheets_perfect = 0

    for i in range(num_sheets):
        t = str(["A", "B", "C", "D"][i % 4])
        params = SyntheticTransformParams(
            rotation_deg=float(rng.uniform(-15.0, 15.0)),
            scale_factor=float(rng.uniform(0.72, 0.94)),
            perspective_jitter_px=float(rng.uniform(12.0, 44.0)),
            shadow_strength=float(rng.uniform(0.05, 0.50)),
            shadow_angle_deg=float(rng.uniform(0.0, 180.0)),
            gaussian_noise_sigma=float(rng.uniform(2.0, 10.0)),
            salt_pepper_prob=float(rng.choice([0.0, 0.0, 0.004])),
            mark_intensity=int(rng.integers(25, 75)),
        )
        sample = create_synthetic_sample(
            f"bench_{i:02d}",
            exam_config=exam_cfg,
            exam_type=t,
            params=params,
            layout=layout,
            seed=1000 + i * 31,
        )
        res = evaluate_sample(sample, pipeline)
        if res["marker_detected"]:
            markers_ok += 1
            corner_rmses.append(float(res["corner_rmse_px"]))
            if res["type_correct"]:
                types_ok += 1
            if res["sheet_perfect"]:
                sheets_perfect += 1
            q_accs.append(float(res["question_accuracy"]))
            b_accs.append(float(res["bubble_accuracy"]))
            all_bubble_pairs.extend(res["bubble_pairs"])
            all_question_pairs.extend(res["question_pairs"])

    # 1. Bubble 2x2 confusion matrix: 0=Vazia, 1=Marcada
    b_cm = np.zeros((2, 2), dtype=int)
    for t_b, p_b in all_bubble_pairs:
        b_cm[t_b, p_b] += 1

    # 2. Question 7x7 confusion matrix: A, B, C, D, E, Branco, Dupla
    classes = ["A", "B", "C", "D", "E", "Branco", "Dupla"]
    cls_idx = {c: idx for idx, c in enumerate(classes)}
    q_cm = np.zeros((len(classes), len(classes)), dtype=int)
    for t_q, p_q in all_question_pairs:
        q_cm[cls_idx[t_q], cls_idx[p_q]] += 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 5.0), dpi=300)

    # Plot bubble confusion matrix
    im1 = ax1.imshow(b_cm, cmap="Blues")
    ax1.set_title("(a) Matriz de Confusão por Bolha (N = 7.500)", fontsize=10.5, fontweight="bold")
    ax1.set_xticks([0, 1], labels=["Lida: Vazia", "Lida: Marcada"])
    ax1.set_yticks([0, 1], labels=["Real: Vazia", "Real: Marcada"])
    for r in range(2):
        for c in range(2):
            val = b_cm[r, c]
            color = "white" if val > b_cm.max() * 0.5 else "black"
            ax1.text(c, r, f"{val}", ha="center", va="center", color=color, fontsize=11, fontweight="bold")

    # Plot question confusion matrix
    im2 = ax2.imshow(q_cm, cmap="Blues")
    ax2.set_title("(b) Matriz de Confusão por Questão (N = 1.500)", fontsize=10.5, fontweight="bold")
    ax2.set_xticks(range(len(classes)), labels=classes)
    ax2.set_yticks(range(len(classes)), labels=classes)
    ax2.set_xlabel("Classe Predita pelo Sistema", fontsize=9.5)
    ax2.set_ylabel("Classe Verdadeira (Ground Truth)", fontsize=9.5)
    for r in range(len(classes)):
        for c in range(len(classes)):
            val = q_cm[r, c]
            color = "white" if val > q_cm.max() * 0.5 else "black"
            ax2.text(c, r, f"{val}", ha="center", va="center", color=color, fontsize=9.5, fontweight="bold")

    plt.tight_layout()
    fig.savefig(fig_dir / "fig_matriz_confusao.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    tn, fp, fn, tp = int(b_cm[0, 0]), int(b_cm[0, 1]), int(b_cm[1, 0]), int(b_cm[1, 1])
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)

    return {
        "num_sheets": num_sheets,
        "marker_detection_rate_pct": round(100.0 * markers_ok / num_sheets, 2),
        "mean_corner_rmse_px": round(float(np.mean(corner_rmses)), 3),
        "exam_type_accuracy_pct": round(100.0 * types_ok / max(1, markers_ok), 2),
        "question_accuracy_pct": round(100.0 * float(np.mean(q_accs)), 2),
        "bubble_accuracy_pct": round(100.0 * float(np.mean(b_accs)), 2),
        "sheet_exact_match_pct": round(100.0 * sheets_perfect / num_sheets, 2),
        "bubble_confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "bubble_precision_pct": round(100.0 * precision, 2),
        "bubble_recall_pct": round(100.0 * recall, 2),
        "bubble_f1_pct": round(100.0 * f1, 2),
    }


def run_robustness_curves(
    exam_cfg: ExamConfig,
    fig_dir: Path,
) -> Dict[str, object]:
    """Sweeps rotation, scale, Gaussian noise, and salt-and-pepper noise to plot robustness curves."""
    layout = get_default_layout()
    pipe_gauss = OMRPipeline(PipelineConfig(denoise_method="gaussian"), layout, exam_cfg)
    pipe_median = OMRPipeline(PipelineConfig(denoise_method="median"), layout, exam_cfg)

    # 1. Rotation angle sweep (degrees)
    rot_angles = [0, 5, 10, 15, 20, 25]
    rot_accs: List[float] = []
    for ang in rot_angles:
        sample = create_synthetic_sample(
            f"rot_{ang}",
            exam_cfg,
            params=SyntheticTransformParams(rotation_deg=float(ang), scale_factor=0.84, perspective_jitter_px=15, shadow_strength=0.15),
            layout=layout,
            seed=2100 + ang,
        )
        res = evaluate_sample(sample, pipe_gauss)
        rot_accs.append(float(res["question_accuracy"]) * 100.0)

    # 2. Scale factor sweep
    scales = [0.55, 0.65, 0.75, 0.85, 0.95]
    scale_accs: List[float] = []
    for idx, sc in enumerate(scales):
        sample = create_synthetic_sample(
            f"scale_{idx}",
            exam_cfg,
            params=SyntheticTransformParams(rotation_deg=3.0, scale_factor=sc, perspective_jitter_px=15, shadow_strength=0.15),
            layout=layout,
            seed=2200 + idx,
        )
        res = evaluate_sample(sample, pipe_gauss)
        scale_accs.append(float(res["question_accuracy"]) * 100.0)

    # 3. Gaussian noise sigma sweep
    sigmas = [0, 5, 10, 15, 25, 35]
    gauss_noise_accs: List[float] = []
    for sig in sigmas:
        sample = create_synthetic_sample(
            f"gnoise_{sig}",
            exam_cfg,
            params=SyntheticTransformParams(rotation_deg=3.0, scale_factor=0.86, gaussian_noise_sigma=float(sig), shadow_strength=0.15),
            layout=layout,
            seed=2300 + sig,
        )
        res = evaluate_sample(sample, pipe_gauss)
        gauss_noise_accs.append(float(res["question_accuracy"]) * 100.0)

    # 4. Salt-and-pepper noise probability sweep: Gaussian filter vs. Median filter (Week 4)
    sp_probs = [0.0, 0.01, 0.02, 0.04, 0.06, 0.08]
    sp_acc_gauss: List[float] = []
    sp_acc_median: List[float] = []
    for idx, p_sp in enumerate(sp_probs):
        sample = create_synthetic_sample(
            f"sp_{idx}",
            exam_cfg,
            params=SyntheticTransformParams(
                rotation_deg=2.0,
                scale_factor=0.88,
                perspective_jitter_px=15,
                shadow_strength=0.15,
                gaussian_noise_sigma=2.0,
                salt_pepper_prob=p_sp,
            ),
            layout=layout,
            seed=2400 + idx,
        )
        r_g = evaluate_sample(sample, pipe_gauss)
        r_m = evaluate_sample(sample, pipe_median)
        sp_acc_gauss.append(float(r_g["question_accuracy"]) * 100.0)
        sp_acc_median.append(float(r_m["question_accuracy"]) * 100.0)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), dpi=300)

    axes[0, 0].plot(rot_angles, rot_accs, "o-", color="#1f4e79", linewidth=2.2)
    axes[0, 0].set_title("(a) Robustez à Rotação no Plano", fontsize=10, fontweight="bold")
    axes[0, 0].set_xlabel("Ângulo de rotação (graus)")
    axes[0, 0].set_ylabel("Acurácia por questão (%)")
    axes[0, 0].set_ylim(50, 103)
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)

    axes[0, 1].plot([s * 100 for s in scales], scale_accs, "s-", color="#27ae60", linewidth=2.2)
    axes[0, 1].set_title("(b) Robustez ao Fator de Escala", fontsize=10, fontweight="bold")
    axes[0, 1].set_xlabel("Escala da folha na imagem (%)")
    axes[0, 1].set_ylabel("Acurácia por questão (%)")
    axes[0, 1].set_ylim(50, 103)
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)

    axes[1, 0].plot(sigmas, gauss_noise_accs, "^-", color="#8e44ad", linewidth=2.2)
    axes[1, 0].set_title("(c) Robustez ao Ruído Gaussiano", fontsize=10, fontweight="bold")
    axes[1, 0].set_xlabel("Desvio padrão do ruído σ (níveis de cinza)")
    axes[1, 0].set_ylabel("Acurácia por questão (%)")
    axes[1, 0].set_ylim(50, 103)
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)

    sp_pct = [p * 100 for p in sp_probs]
    axes[1, 1].plot(sp_pct, sp_acc_gauss, "o--", color="#c0392b", linewidth=2.0, label="Filtro Gaussiano 5×5")
    axes[1, 1].plot(sp_pct, sp_acc_median, "s-", color="#1f4e79", linewidth=2.4, label="Filtro da Mediana 5×5")
    axes[1, 1].set_title("(d) Ruído Sal-e-Pimenta: Gaussiano vs. Mediana", fontsize=10, fontweight="bold")
    axes[1, 1].set_xlabel("Densidade de ruído sal-e-pimenta (%)")
    axes[1, 1].set_ylabel("Acurácia por questão (%)")
    axes[1, 1].set_ylim(40, 103)
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)
    axes[1, 1].legend(fontsize=8.5, loc="lower left")

    plt.tight_layout()
    fig.savefig(fig_dir / "fig_curvas_robustez.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "rotation_deg": rot_angles,
        "rotation_acc_pct": [round(v, 2) for v in rot_accs],
        "scale_factors": scales,
        "scale_acc_pct": [round(v, 2) for v in scale_accs],
        "gaussian_noise_sigma": sigmas,
        "gaussian_noise_acc_pct": [round(v, 2) for v in gauss_noise_accs],
        "salt_pepper_probs": sp_probs,
        "salt_pepper_acc_gaussian_pct": [round(v, 2) for v in sp_acc_gauss],
        "salt_pepper_acc_median_pct": [round(v, 2) for v in sp_acc_median],
    }


def main() -> int:
    fig_dir = ROOT_DIR / "docs" / "relatorio" / "figuras"
    fig_dir.mkdir(parents=True, exist_ok=True)
    exam_cfg = create_default_exam_config(seed=2026)

    print("[1/4] Gerando figura qualitativa das etapas do pipeline...")
    save_pipeline_stages_figure(exam_cfg, fig_dir)

    print("[2/4] Executando experimento comparativo Otsu vs. Limiar Adaptativo...")
    otsu_vs_adap = run_otsu_vs_adaptive_experiment(exam_cfg, fig_dir)

    print("[3/4] Executando avaliacao quantitativa em lote (30 folhas, 1.500 questoes, 7.500 bolhas)...")
    bench_metrics = run_benchmark_and_confusion_matrix(exam_cfg, fig_dir, num_sheets=30)

    print("[4/4] Gerando curvas de robustez (rotacao, escala, ruido Gaussiano e sal-e-pimenta)...")
    robust_metrics = run_robustness_curves(exam_cfg, fig_dir)

    summary = {
        "benchmark": bench_metrics,
        "otsu_vs_adaptive": otsu_vs_adap,
        "robustness": robust_metrics,
    }
    metrics_path = ROOT_DIR / "docs" / "relatorio" / "metricas_avaliacao.json"
    metrics_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== RESUMO QUANTITATIVO ===")
    print(f"- Taxa de deteccao dos 4 marcadores: {bench_metrics['marker_detection_rate_pct']:.2f}%")
    print(f"- Erro medio de localizacao (RMSE):  {bench_metrics['mean_corner_rmse_px']:.2f} px")
    print(f"- Acuracia de leitura por bolha:     {bench_metrics['bubble_accuracy_pct']:.2f}% (F1 = {bench_metrics['bubble_f1_pct']:.2f}%)")
    print(f"- Acuracia de leitura por questao:   {bench_metrics['question_accuracy_pct']:.2f}%")
    print(f"- Acuracia de leitura de tipo (A-D): {bench_metrics['exam_type_accuracy_pct']:.2f}%")
    print(f"- Figuras salvas em:                 {fig_dir}")
    print(f"- Metricas JSON salvas em:           {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
