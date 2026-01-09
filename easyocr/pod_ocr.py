"""
POD (Proof of Delivery) OCR Module - EasyOCR Version

Extracts street and unit numbers from delivery photos using EasyOCR.
Targets >95% accuracy with confidence-based filtering.
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
import gc

# Lazy-load OCR to avoid import overhead when not needed
_ocr_instance = None


def get_ocr():
    """Lazily initialize EasyOCR instance."""
    global _ocr_instance
    if _ocr_instance is None:
        import easyocr
        # Initialize EasyOCR reader for English
        # gpu=True/False will be automatically determined or can be explicit
        _ocr_instance = easyocr.Reader(['en'])
    return _ocr_instance


def cleanup_ocr():
    """Release OCR reader and free memory."""
    global _ocr_instance
    if _ocr_instance is not None:
        del _ocr_instance
        _ocr_instance = None
    gc.collect()


def preprocess_image(image_path: str):
    """
    Preprocess image for better OCR accuracy.
    
    Args:
        image_path: Path to the image file.
        
    Returns:
        Preprocessed grayscale image with enhanced contrast.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Enhance contrast using histogram equalization
    enhanced = cv2.equalizeHist(gray)
    return enhanced


def parse_street_number(text: str) -> Optional[str]:
    """
    Parse street number from OCR text.
    
    Matches patterns like: 123, 1234, 12345, 123A, 1234B
    """
    match = re.match(r'^(\d{1,5}[A-Z]?)$', text.strip(), re.IGNORECASE)
    return match.group(1) if match else None


def parse_unit_number(text: str) -> Optional[str]:
    """
    Parse unit/apartment number from OCR text.
    
    Matches patterns like: Apt 1, Unit 2A, #3, Suite 100
    """
    match = re.search(r'(Apt|Unit|#|Suite)\s*(\d+[A-Z]?)', text, re.IGNORECASE)
    return match.group(0) if match else None


def parse_street_name(text: str) -> Optional[str]:
    """
    Parse street name from OCR text.
    
    Matches common street name patterns (alphabetic words that could be street names).
    Excludes common non-street words.
    """
    # Common street type suffixes
    street_types = r'(Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Court|Ct|Boulevard|Blvd|Way|Place|Pl|Circle|Cir|Close|Terrace|Ter|Trail|Trl|Park|Parkway|Pkwy)'
    
    # Check if this text contains a street type
    if re.search(street_types, text, re.IGNORECASE):
        return text.strip()
    
    # Check for standalone proper nouns (potential street names)
    # Must be alphabetic, start with uppercase, 3+ chars
    if re.match(r'^[A-Z][a-z]{2,}$', text.strip()):
        # Exclude common non-street words
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
        confidence_threshold: Minimum OCR confidence (default 0.95).
        
    Returns:
        Dictionary with street_number, street_name, unit_number, confidence, and processing_time.
    """
    start_time = time.time()
    try:
        ocr = get_ocr()
        predict_start = time.time()
        
        # EasyOCR readtext returns list of (bbox, text, prob)
        result = ocr.readtext(image_path)
        
        processing_time = time.time() - predict_start
        
        if not result or len(result) == 0:
            return {
                'image_path': image_path,
                'street_number': None, 
                'street_name': None, 
                'unit_number': None, 
                'confidence': 0.0,
                'processing_time': round(processing_time, 3)
            }
        
        # Extract text with confidence above threshold
        extracted_texts = []
        max_confidence = 0.0
        
        for item in result:
            # EasyOCR result item structure: (bbox, text, prob)
            # bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            text = item[1]
            prob = item[2]
            
            if prob >= confidence_threshold:
                extracted_texts.append(text)
            max_confidence = max(max_confidence, prob)
        
        # Find street number, street name, and unit number
        street_number = None
        street_name = None
        unit_number = None
        street_names = []  # Collect all potential street name parts
        
        for text in extracted_texts:
            if not street_number:
                street_number = parse_street_number(text)
            if not unit_number:
                unit_number = parse_unit_number(text)
            # Collect street name parts
            name = parse_street_name(text)
            if name:
                street_names.append(name)
        
        # Combine street name parts
        if street_names:
            street_name = ' '.join(street_names)
        
        return {
            'image_path': image_path,
            'filename': os.path.basename(image_path),
            'street_number': street_number,
            'street_name': street_name,
            'unit_number': unit_number,
            'confidence': round(max_confidence, 3),
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
    """
    Process a single image (for multiprocessing).
    
    Args:
        image_path: Path to the image.
        
    Returns:
        Dictionary mapping image path to extraction result.
    """
    try:
        return {image_path: extract_numbers(image_path)}
    except Exception as e:
        return {image_path: {'error': str(e)}}


def process_batch(image_dir: str, output_file: str = 'results.json') -> dict:
    """
    Process all images in a directory (single-threaded).
    
    Args:
        image_dir: Directory containing images.
        output_file: Output JSON file path.
        
    Returns:
        Dictionary of results.
    """
    image_paths = list(Path(image_dir).glob('*.jpg')) + list(Path(image_dir).glob('*.jpeg')) + list(Path(image_dir).glob('*.png'))
    
    results = {}
    for path in image_paths:
        results[path.name] = extract_numbers(str(path))
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def process_batch_parallel(image_dir: str, output_file: str = 'results.json', workers: int = 8) -> dict:
    """
    Process all images in a directory using multiprocessing.
    
    Args:
        image_dir: Directory containing images.
        output_file: Output JSON file path.
        workers: Number of parallel workers.
        
    Returns:
        Dictionary of results.
    """
    image_paths = [str(p) for p in Path(image_dir).glob('*.jpg')]
    image_paths += [str(p) for p in Path(image_dir).glob('*.jpeg')]
    image_paths += [str(p) for p in Path(image_dir).glob('*.png')]
    
    with Pool(workers) as pool:
        results_list = pool.map(process_single_image, image_paths)
    
    # Merge results
    merged = {}
    for res in results_list:
        for k, v in res.items():
            merged[Path(k).name] = v
    
    with open(output_file, 'w') as f:
        json.dump(merged, f, indent=2)
    
    return merged


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description='POD OCR (EasyOCR) - Extract street/unit numbers from delivery photos'
    )
    parser.add_argument(
        'input',
        help='Path to image file or directory of images'
    )
    parser.add_argument(
        '-o', '--output',
        default='results.json',
        help='Output JSON file (default: results.json)'
    )
    parser.add_argument(
        '-w', '--workers',
        type=int,
        default=1,
        help='Number of parallel workers for batch processing (default: 1, use higher values with caution on memory-limited systems)'
    )
    parser.add_argument(
        '-c', '--confidence',
        type=float,
        default=0.95,
        help='Minimum confidence threshold (default: 0.95)'
    )
    
    args = parser.parse_args()
    input_path = Path(args.input)
    
    try:
        if input_path.is_file():
            # Single image
            result = extract_numbers(str(input_path), args.confidence)
            print(json.dumps({input_path.name: result}, indent=2))
        elif input_path.is_dir():
            # Batch processing
            if args.workers > 1:
                results = process_batch_parallel(str(input_path), args.output, args.workers)
            else:
                results = process_batch(str(input_path), args.output)
            print(f"Processed {len(results)} images. Results saved to {args.output}")
        else:
            print(f"Error: {args.input} does not exist")
            exit(1)
    finally:
        # Always cleanup to free memory
        cleanup_ocr()


if __name__ == '__main__':
    main()
