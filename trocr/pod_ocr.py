"""
POD (Proof of Delivery) OCR Module - TrOCR Version

Extracts street and unit numbers from delivery photos using Microsoft's TrOCR.
Targets high accuracy for printed text using Transformer-based architecture.
"""

import cv2
import re
import json
import argparse
import time
import os
from pathlib import Path
from multiprocessing import Pool
from typing import Optional, Tuple

# Lazy-load models
_processor = None
_model = None


def get_ocr():
    """Lazily initialize TrOCR model and processor."""
    global _processor, _model
    if _model is None:
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        # Use the base printed model which is best suited for receipts/labels
        model_name = "microsoft/trocr-base-printed"
        
        try:
            _processor = TrOCRProcessor.from_pretrained(model_name)
            _model = VisionEncoderDecoderModel.from_pretrained(model_name)
            
            # Use GPU if available
            if torch.cuda.is_available():
                _model.to("cuda")
        except Exception as e:
            raise RuntimeError(f"Failed to load TrOCR model: {e}")

    return _processor, _model


def preprocess_image(image_path: str):
    """
    Load image for TrOCR.
    TrOCRProcessor handles the actual resizing/normalization.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    # Convert BGR to RGB as Transformers expects RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def parse_street_number(text: str) -> Optional[str]:
    """Parse street number from OCR text."""
    match = re.match(r'^(\d{1,5}[A-Z]?)$', text.strip(), re.IGNORECASE)
    return match.group(1) if match else None


def parse_unit_number(text: str) -> Optional[str]:
    """Parse unit/apartment number from OCR text."""
    match = re.search(r'(Apt|Unit|#|Suite)\s*(\d+[A-Z]?)', text, re.IGNORECASE)
    return match.group(0) if match else None


def parse_street_name(text: str) -> Optional[str]:
    """Parse street name from OCR text."""
    street_types = r'(Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Court|Ct|Boulevard|Blvd|Way|Place|Pl|Circle|Cir|Close|Terrace|Ter|Trail|Trl|Park|Parkway|Pkwy)'
    
    if re.search(street_types, text, re.IGNORECASE):
        return text.strip()
    
    if re.match(r'^[A-Z][a-z]{2,}$', text.strip()):
        excluded = {'The', 'Dear', 'Customer', 'Proof', 'Delivery', 'Tracking', 'Number', 
                    'Weight', 'Service', 'Shipped', 'Billed', 'Delivered', 'Left', 
                    'Reference', 'Please', 'Print', 'Sincerely', 'Front', 'Door'}
        if text.strip() not in excluded:
            return text.strip()
    return None


def extract_numbers(image_path: str, confidence_threshold: float = 0.95) -> dict:
    """
    Extract street and unit numbers from an image.
    
    Args:
        image_path: Path to the delivery photo.
        confidence_threshold: Not applicable for standard TrOCR generation, kept for API compatibility.
        
    Returns:
        Dictionary with results.
    """
    start_time = time.time()
    try:
        import torch
        processor, model = get_ocr()
        predict_start = time.time()
        
        # Load and process image
        image = preprocess_image(image_path)
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        
        if torch.cuda.is_available():
            pixel_values = pixel_values.to("cuda")

        # Generate text
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        processing_time = time.time() - predict_start
        
        # TrOCR generates a single string. We need to split it to mimic finding "instances" of text.
        # This is surprisingly effective if the model reads the text line by line or space separated.
        extracted_texts = generated_text.split()
        
        street_number = None
        street_name = None
        unit_number = None
        street_names = []
        
        # Also check the full text for unit number in case "Apt 1" was generated as "Apt 1"
        if not unit_number:
            unit_number = parse_unit_number(generated_text)

        for text in extracted_texts:
            clean_text = text.strip('.,;:()[]"')
            
            if not street_number:
                street_number = parse_street_number(clean_text)
            
            # If we didn't find unit number in full text (maybe split by newline?), check parts
            if not unit_number:
                unit_number = parse_unit_number(clean_text)
                
            name = parse_street_name(clean_text)
            if name:
                street_names.append(name)
        
        if street_names:
            street_name = ' '.join(street_names)
        
        return {
            'image_path': image_path,
            'filename': os.path.basename(image_path),
            'street_number': street_number,
            'street_name': street_name,
            'unit_number': unit_number,
            'confidence': 1.0, # TrOCR doesn't give easy word-level confidence
            'processing_time': round(processing_time, 3)
        }
    except Exception as e:
        return {
            'image_path': image_path,
            'filename': os.path.basename(image_path),
            'street_number': None,
            'street_name': None,
            'unit_number': None,
            'confidence': 0.0,
            'error': str(e),
            'processing_time': round(time.time() - start_time, 3)
        }


def process_single_image(image_path: str) -> dict:
    """Process a single image."""
    try:
        return {image_path: extract_numbers(image_path)}
    except Exception as e:
        return {image_path: {'error': str(e)}}


def process_batch(image_dir: str, output_file: str = 'results.json') -> dict:
    """Process all images in a directory (single-threaded)."""
    image_paths = list(Path(image_dir).glob('*.jpg')) + list(Path(image_dir).glob('*.jpeg')) + list(Path(image_dir).glob('*.png'))
    results = {}
    for path in image_paths:
        results[path.name] = extract_numbers(str(path))
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    return results


def process_batch_parallel(image_dir: str, output_file: str = 'results.json', workers: int = 1) -> dict:
    """
    Process all images in a directory. 
    Parallel processing with heavy models (Transformers) often leads to OOM or race conditions on GPU.
    Defaulting to sequential logic if workers=1 (or implicitly if we want safety).
    """
    image_paths = [str(p) for p in Path(image_dir).glob('*.jpg')]
    image_paths += [str(p) for p in Path(image_dir).glob('*.jpeg')]
    image_paths += [str(p) for p in Path(image_dir).glob('*.png')]
    
    # Using Pool with CUDA is tricky. We'll support it but user must be careful.
    if workers > 1:
        with Pool(workers) as pool:
            results_list = pool.map(process_single_image, image_paths)
    else:
        results_list = [process_single_image(p) for p in image_paths]
    
    merged = {}
    for res in results_list:
        for k, v in res.items():
            merged[Path(k).name] = v
            
    with open(output_file, 'w') as f:
        json.dump(merged, f, indent=2)
        
    return merged


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description='POD OCR (TrOCR)')
    parser.add_argument('input', help='Path to image file or directory')
    parser.add_argument('-o', '--output', default='results.json', help='Output JSON file')
    parser.add_argument('-w', '--workers', type=int, default=1, help='Workers (Default 1)')
    parser.add_argument('-c', '--confidence', type=float, default=0.95, help='Unused for TrOCR')
    
    args = parser.parse_args()
    input_path = Path(args.input)
    
    if input_path.is_file():
        result = extract_numbers(str(input_path), args.confidence)
        print(json.dumps({input_path.name: result}, indent=2))
    elif input_path.is_dir():
        if args.workers > 1:
            results = process_batch_parallel(str(input_path), args.output, args.workers)
        else:
            results = process_batch(str(input_path), args.output)
        print(f"Processed {len(results)} images. Results saved to {args.output}")
    else:
        print(f"Error: {args.input} does not exist")
        exit(1)


if __name__ == '__main__':
    main()
