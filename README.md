# OMR Exam Grader

Automatic grading of multiple-choice answer sheets using Optical Mark Recognition (OMR).
Course project for Digital Image Processing (PDI), UTFPR, 2026-2.

A sheet (50 questions x 5 options, up to 4 exam types) is photographed with a smartphone;
the system rectifies the page, reads the marks, compares them with the answer key and outputs a grade.

## Pipeline
1. Acquisition (smartphone photo)
2. Preprocessing (grayscale, denoising, normalization)
3. Binarization (Otsu / adaptive threshold)
4. Corner marker detection (morphology + connected components)
5. Rectification (homography)
6. Bubble grid sampling (50 x 5)
7. Mark decision (fill ratio; blank / double marks)
8. Exam type reading
9. Grading (answer keys, per-question weights)

## Stack
Python 3, OpenCV, NumPy, Matplotlib, Pillow, ReportLab, pytest (Streamlit optional for the demo UI).

## Setup
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Layout
- `src/omr/` pipeline code
- `tools/` answer sheet generation and utilities
- `data/` test photos and ground truth
- `tests/` automated tests
- `docs/` project overview, report and slides

Authors: Romulo and Guilherme Oliver.
