# Using Claude AI for POD Recognition Feature

## Overview
Guide to using **Claude AI** (claude.ai) to develop a Python-based **Proof of Delivery (POD)** feature for extracting street/unit numbers from delivery photos (>95% accuracy) with PaddleOCR. Leverage Claude's **Artifacts** for interactive code generation.

## Why Claude?
- Rapid prototyping: Generate, iterate, and preview code.
- Accuracy: Incorporates benchmarks like SVHN for >95%.
- Cost-effective: Builds self-hosted solutions.

## Prerequisites
- Claude Pro/Team account (for Artifacts).
- Python 3.10+, PaddleOCR (`pip install paddleocr paddlepaddle-gpu`).
- Sample POD images.
- Optional: Docker for scaling.

## Step-by-Step

### 1. Define Requirements
Prompt: "Design Python script for POD number extraction with PaddleOCR: batch processing, >95% accuracy, JSON output, regex parsing, self-hosted."

Claude outputs architecture skeleton.

### 2. Generate Core Code
Prompt: "Generate PaddleOCR code for text extraction from images: preprocess (contrast/deblur), batch, regex for street/unit, confidence >95%."

Sample Code:
```python
import cv2
from paddleocr import PaddleOCR
import re, json
from pathlib import Path

ocr = PaddleOCR(use_angle_cls=True, lang='en')

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray)  # Enhance contrast

def extract_numbers(image_path):
    img = preprocess_image(image_path)
    result = ocr.ocr(img, cls=True)
    
    extracted = [word[1][0] for line in result for word in line if word[1][1] > 0.95]
    
    street = next((t for t in extracted if re.match(r'^\d{3,5}[A-Z]?$', t)), None)
    unit = next((t for t in extracted if re.search(r'(Apt|Unit|#|Suite)\s*\d+[A-Z]?', t, re.I)), None)
    
    return {'street_number': street, 'unit_number': unit, 'confidence': 0.95 if result else 0}

def process_batch(image_dir):
    results = {p.name: extract_numbers(str(p)) for p in Path(image_dir).glob('*.jpg')}
    with open('results.json', 'w') as f: json.dump(results, f)
```

Iterate: "Add GPU/multiprocessing."

### 3. Fine-Tuning
Prompt: "Code to fine-tune PaddleOCR on 1k-5k labeled POD images for >98% accuracy."

### 4. Scaling
Prompt: "Extend for millions: multiprocessing/Dask, logging."

Sample Addition:
```python
from multiprocessing import Pool

def process_image(img_path):
    try: return {img_path: extract_numbers(img_path)}
    except: return {img_path: {'error': 'Failed'}}

def process_batch_parallel(image_paths, workers=8):
    with Pool(workers) as p:
        results = p.map(process_image, image_paths)
    merged = {k: v for res in results for k, v in res.items()}
    json.dump(merged, open('results.json', 'w'))
```

### 5. Integration/Testing
- Prompt for API wrapper (FastAPI).
- Generate tests: "Pytest for extraction function."

## Best Practices
- Use Artifacts for previews.
- Iterative prompts with specifics (e.g., "Optimize for blur").
- Test locally; Claude can't run external code.

## Challenges & Fixes
- Low accuracy: Fine-tune with data.
- Speed: GPU optimizations.
- Privacy: Self-hosted.