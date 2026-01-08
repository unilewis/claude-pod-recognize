# POD Recognition Feature Implementation Plan

## Goal Description

Develop a Python-based **Proof of Delivery (POD)** feature for extracting street/unit numbers from delivery photos using PaddleOCR, aiming for >95% accuracy.

## User Review Required

- **Prerequisites**: The user needs `paddleocr` and `paddlepaddle-gpu` (or cpu version) installed. I will create a `requirements.txt` but the user might need to install system dependencies for opencv.
- **Hardware**: GPU is recommended for speed but CPU works for testing.

## Proposed Changes

### Configuration

#### [NEW] [requirements.txt](file:///Users/llewis/Code/claude-pod-recognize/requirements.txt)

- `paddleocr`
- `paddlepaddle` (using cpu version for compatibility unless specified)
- `opencv-python`
- `pytest`

### Core Logic

#### [NEW] [pod_ocr.py](file:///Users/llewis/Code/claude-pod-recognize/pod_ocr.py)

- **`preprocess_image(image_path)`**: Enhance contrast/deblur.
- **`extract_numbers(image_path)`**: Run PaddleOCR and parse results.
- **`process_batch_parallel(image_dir)`**: Handle multiple files using multiprocessing.

### Testing

#### [NEW] [test_pod_ocr.py](file:///Users/llewis/Code/claude-pod-recognize/test_pod_ocr.py)

- Unit tests for regex parsing.
- Mock tests for OCR (since we might not have the model downloaded or images).

## Verification Plan

### Automated Tests

- Run `pytest test_pod_ocr.py`.

### Manual Verification

- If the user provides an image, run `python pod_ocr.py <image_path>`.
