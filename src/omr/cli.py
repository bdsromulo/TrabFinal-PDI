"""Command-Line Interface (CLI) for grading single answer sheets or batch folders."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import List, Optional

import cv2
import numpy as np

from .grade import ExamConfig, create_default_exam_config
from .pipeline import OMRPipeline, PipelineConfig


def _imwrite_unicode(path: Path, image: np.ndarray) -> None:
    """Writes an OpenCV image supporting Unicode file paths on Windows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".png"
    ok, buf = cv2.imencode(ext, image)
    if ok:
        buf.tofile(str(path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omr-grade",
        description="OMR Exam Grader — Correcao automatica de folhas de respostas (PDI UTFPR).",
    )
    parser.add_argument(
        "input",
        type=str,
        help="Caminho para uma foto de folha de respostas (.png, .jpg) ou pasta com fotos.",
    )
    parser.add_argument(
        "--keys",
        type=str,
        default=None,
        help="Caminho para o arquivo JSON com os gabaritos (tipos A-D) e pesos das questoes.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Diretorio onde salvar imagens anotadas, etapas de debug e CSV de notas.",
    )
    parser.add_argument(
        "--binarize",
        choices=["adaptive", "otsu"],
        default="adaptive",
        help="Metodo de binarizacao: 'adaptive' (padrao) ou 'otsu'.",
    )
    parser.add_argument(
        "--denoise",
        choices=["gaussian", "median", "bilateral", "none"],
        default="gaussian",
        help="Filtro de suavizacao espacial antes da binarizacao.",
    )
    parser.add_argument(
        "--no-illumination-norm",
        action="store_true",
        help="Desativa a normalizacao morfologica de fundo/iluminacao.",
    )
    parser.add_argument(
        "--save-stages",
        action="store_true",
        help="Salva todas as imagens intermediarias do pipeline (cinza, binaria, marcadores, retificada).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERRO] Caminho nao encontrado: {input_path}", file=sys.stderr)
        return 1

    if args.keys:
        exam_config = ExamConfig.load_json(args.keys)
    else:
        default_keys_path = Path(__file__).resolve().parents[2] / "data" / "answer_keys" / "default_keys.json"
        if default_keys_path.exists():
            exam_config = ExamConfig.load_json(default_keys_path)
        else:
            exam_config = create_default_exam_config()

    cfg = PipelineConfig(
        denoise_method=args.denoise,
        normalize_illumination=not args.no_illumination_norm,
        binarize_method=args.binarize,
    )
    pipeline = OMRPipeline(config=cfg, exam_config=exam_config)

    if input_path.is_dir():
        image_files = sorted(
            p for p in input_path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        )
    else:
        image_files = [input_path]

    if not image_files:
        print(f"[ERRO] Nenhuma imagem encontrada em: {input_path}", file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_rows: List[dict] = []
    for img_path in image_files:
        stem = img_path.stem
        try:
            grading, artifacts = pipeline.process_file(img_path)
            graded_out = out_dir / f"{stem}_graded.png"
            _imwrite_unicode(graded_out, artifacts.graded_image)

            if args.save_stages:
                stages_dir = out_dir / f"{stem}_stages"
                _imwrite_unicode(stages_dir / "01_original.png", artifacts.original_bgr)
                _imwrite_unicode(stages_dir / "02_gray.png", artifacts.preprocess.gray)
                _imwrite_unicode(stages_dir / "03_normalized.png", artifacts.preprocess.normalized)
                _imwrite_unicode(stages_dir / "04_binary.png", artifacts.preprocess.binary)
                _imwrite_unicode(stages_dir / "05_markers.png", artifacts.markers.debug_image)
                _imwrite_unicode(stages_dir / "06_rectified_gray.png", artifacts.rectification.rectified_gray)
                _imwrite_unicode(stages_dir / "07_rectified_binary.png", artifacts.rectification.rectified_binary)
                _imwrite_unicode(stages_dir / "08_roi_grid.png", artifacts.rectification.grid_overlay)
                _imwrite_unicode(stages_dir / "09_graded.png", artifacts.graded_image)

            json_out = out_dir / f"{stem}_report.json"
            json_out.write_text(
                json.dumps(grading.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            print(
                f"[OK] {img_path.name} -> Tipo: {grading.used_exam_type} | "
                f"Nota: {grading.total_score:.2f}/{grading.max_score:.2f} ({grading.score_normalized_10:.1f}/10) | "
                f"Acertos: {grading.num_correct} | Erros: {grading.num_wrong} | "
                f"Brancos: {grading.num_blank} | Duplas: {grading.num_multiple}"
            )
            csv_rows.append(
                {
                    "arquivo": img_path.name,
                    "status": "OK",
                    "tipo_detectado": grading.detected_exam_type or "",
                    "tipo_usado": grading.used_exam_type or "",
                    "nota": f"{grading.total_score:.2f}",
                    "nota_max": f"{grading.max_score:.2f}",
                    "nota_10": f"{grading.score_normalized_10:.2f}",
                    "acertos": grading.num_correct,
                    "erros": grading.num_wrong,
                    "brancos": grading.num_blank,
                    "duplas": grading.num_multiple,
                    "limiar_calibrado": f"{grading.calibrated_threshold:.4f}",
                }
            )
        except Exception as exc:
            print(f"[FALHA] {img_path.name}: {exc}", file=sys.stderr)
            csv_rows.append(
                {
                    "arquivo": img_path.name,
                    "status": f"ERRO: {exc}",
                    "tipo_detectado": "",
                    "tipo_usado": "",
                    "nota": "",
                    "nota_max": "",
                    "nota_10": "",
                    "acertos": "",
                    "erros": "",
                    "brancos": "",
                    "duplas": "",
                    "limiar_calibrado": "",
                }
            )

    csv_path = out_dir / "resultados.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "arquivo",
                "status",
                "tipo_detectado",
                "tipo_usado",
                "nota",
                "nota_max",
                "nota_10",
                "acertos",
                "erros",
                "brancos",
                "duplas",
                "limiar_calibrado",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Resumo salvo em: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
