"""
POD (Proof of Delivery) OCR Module - GOT-OCR2.0 Version

Extracts street and unit numbers from delivery photos using GOT-OCR2.0.
Targets >95% accuracy using the state-of-the-art vision-language model.
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
_model = None
_tokenizer = None


def get_ocr():
    """Lazily initialize GOT-OCR2.0 model and tokenizer."""
    global _model, _tokenizer
    if _model is None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        model_name = 'ucaslcl/GOT-OCR2_0'
        
        # Load tokenizer
        _tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Load model
        # Check for GPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Note: GOT-OCR2.0 usually requires CUDA, but we'll attempt to load with what's available.
        # Ideally this should run on a GPU.
        _model = AutoModel.from_pretrained(
            model_name, 
            trust_remote_code=True, 
            low_cpu_mem_usage=True, 
            use_safetensors=True, 
            pad_token_id=_tokenizer.eos_token_id
        )
        
        if torch.cuda.is_available():
            _model = _model.cuda().eval()
        else:
             # Ensure CPU uses float32 to avoid Half/Float mismatch
             _model = _model.float().eval()

    return _model, _tokenizer


def cleanup_ocr():
    """Release model/tokenizer and free memory."""
    global _model, _tokenizer
    
    if _model is not None:
        del _model
        _model = None
    
    if _tokenizer is not None:
        del _tokenizer
        _tokenizer = None
    
    # Clear CUDA cache if available
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    
    gc.collect()


def parse_street_number(text: str) -> Optional[str]:
    """
    Parse street number from text token.
    Matches patterns like: 123, 1234, 12345, 123A, 1234B
    """
    match = re.match(r'^(\d{1,5}[A-Z]?)$', text.strip(), re.IGNORECASE)
    return match.group(1) if match else None


def parse_unit_number(text: str) -> Optional[str]:
    """
    Parse unit/apartment number from text.
    Matches patterns like: Apt 1, Unit 2A, #3, Suite 100
    """
    match = re.search(r'(Apt|Unit|#|Suite)\s*(\d+[A-Z]?)', text, re.IGNORECASE)
    return match.group(0) if match else None


def parse_street_name(text: str) -> Optional[str]:
    """
    Parse street name from text token.
    Matches common street name patterns.
    """
    street_types = r'(Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Court|Ct|Boulevard|Blvd|Way|Place|Pl|Circle|Cir|Close|Terrace|Ter|Trail|Trl|Park|Parkway|Pkwy)'
    
    if re.search(street_types, text, re.IGNORECASE):
        return text.strip()
    
    # Normalize text for matching (handle ALL CAPS)
    normalized = text.strip()
    
    # Match proper words: Title Case or ALL CAPS with 3+ letters
    # Also match ALL CAPS phrases with spaces (e.g., 'NELS ON S')
    if re.match(r'^[A-Z][a-z]{2,}$', normalized) or re.match(r'^[A-Z]{3,}$', normalized) or re.match(r'^[A-Z][A-Z\s]+[A-Z]$', normalized):
        excluded = {'THE', 'DEAR', 'CUSTOMER', 'PROOF', 'DELIVERY', 'TRACKING', 'NUMBER', 
                    'WEIGHT', 'SERVICE', 'SHIPPED', 'BILLED', 'DELIVERED', 'LEFT', 
                    'REFERENCE', 'PLEASE', 'PRINT', 'SINCERELY', 'FRONT', 'DOOR',
                    'The', 'Dear', 'Customer', 'Proof', 'Delivery', 'Tracking', 'Number', 
                    'Weight', 'Service', 'Shipped', 'Billed', 'Delivered', 'Left', 
                    'Reference', 'Please', 'Print', 'Sincerely', 'Front', 'Door'}
        if normalized not in excluded and normalized.upper() not in excluded:
            return normalized
    
    return None


def extract_numbers(image_path: str, confidence_threshold: float = 0.95) -> dict:
    """
    Extract street and unit numbers from an image.
    
    Args:
        image_path: Path to the delivery photo.
        confidence_threshold: Not directly used by GOT-OCR2.0 text generation, but kept for interface consistency.
        
    Returns:
        Dictionary with street_number, street_name, unit_number, confidence, and processing_time.
    """
    start_time = time.time()
    timeout_seconds = 30  # Timeout after 30 seconds
    
    # Use threading for timeout
    import threading
    result_container = [None]
    error_container = [None]
    
    def run_ocr():
        try:
            result_container[0] = _extract_numbers_impl(image_path, confidence_threshold)
        except Exception as e:
            error_container[0] = str(e)
    
    thread = threading.Thread(target=run_ocr)
    thread.start()
    thread.join(timeout=timeout_seconds)
    
    if thread.is_alive():
        # Timeout occurred
        return {
            'image_path': image_path,
            'filename': os.path.basename(image_path),
            'street_number': None,
            'street_name': None,
            'unit_number': None,
            'confidence': 0.0,
            'error': f'Timeout after {timeout_seconds}s',
            'processing_time': round(time.time() - start_time, 3)
        }
    
    if error_container[0]:
        return {
            'image_path': image_path,
            'filename': os.path.basename(image_path),
            'street_number': None,
            'street_name': None,
            'unit_number': None,
            'confidence': 0.0,
            'error': error_container[0],
            'processing_time': round(time.time() - start_time, 3)
        }
    
    return result_container[0]


def _extract_numbers_impl(image_path: str, confidence_threshold: float = 0.95) -> dict:
    """Internal implementation of extract_numbers."""
    start_time = time.time()
    try:
        import torch
        from transformers.cache_utils import DynamicCache
        
        # Monkey patch DynamicCache to support 'seen_tokens' which is used by GOT-OCR code
        # but removed/renamed in recent transformers versions.
        if not hasattr(DynamicCache, 'seen_tokens'):
             DynamicCache.seen_tokens = property(lambda self: self.get_seq_length() if hasattr(self, 'get_seq_length') else 0)
        
        if not hasattr(DynamicCache, 'get_max_length'):
             DynamicCache.get_max_length = lambda self: None

        model, tokenizer = get_ocr()
        predict_start = time.time()
        
        # GOT-OCR2.0 Inference
        # Monkey-patching to handle hardcoded CUDA/Half calls in model.chat()
        
        if model.device.type == 'cpu':
            # Patch torch.Tensor.cuda to return self (no-op)
            orig_cuda = torch.Tensor.cuda
            torch.Tensor.cuda = lambda self, *args, **kwargs: self
            
            # Patch torch.Tensor.half to return float (CPU doesn't support half well)
            orig_half = torch.Tensor.half
            torch.Tensor.half = lambda self, *args, **kwargs: self.float()
            
            # Patch torch.autocast to be a no-op for cuda
            orig_autocast = torch.autocast
            class MockAutocast:
                def __init__(self, device_type, dtype=None, **kwargs):
                    pass
                def __enter__(self):
                    pass
                def __exit__(self, exc_type, exc_val, exc_tb):
                    pass
            torch.autocast = MockAutocast
            
            # Note: Sharpening preprocessing was removed because it caused regressions
            # (e.g., pod6.jpg: '68 ORCHARD CLOSE' became 'ORCHARD CLOSE')
            # The model performs better on original images for most cases.
            
            try:
                res = model.chat(tokenizer, image_path, ocr_type='ocr')
            finally:
                # Restore patches
                torch.Tensor.cuda = orig_cuda
                torch.Tensor.half = orig_half
                torch.autocast = orig_autocast
        else:
            res = model.chat(tokenizer, image_path, ocr_type='ocr')
        
        processing_time = time.time() - predict_start
        
        if not res:
            return {
                'image_path': image_path,
                'street_number': None, 
                'street_name': None, 
                'unit_number': None, 
                'confidence': 1.0, # Model doesn't provide token-level confidence easily in this mode
                'processing_time': round(processing_time, 3)
            }
        
        # GOT-OCR returns full text. We need to tokenize it to use our existing parsers.
        # Splitting by whitespace preserves the logic of "tokens".
        # We might also want to split by newlines if the model returns them.
        extracted_texts = res.split()
        
        street_number = None
        street_name = None
        unit_number = None
        street_names = []
        
        # First pass: combine consecutive ALL CAPS tokens with spaces preserved
        # (e.g., 'NELS', 'ON', 'S' -> 'NELS ON S')
        combined_texts = []
        current_caps_words = []
        for text in extracted_texts:
            clean = text.strip('.,;:()[]"')
            if clean.isupper() and clean.isalpha():
                current_caps_words.append(clean)
            else:
                if current_caps_words:
                    combined_texts.append(' '.join(current_caps_words))
                    current_caps_words = []
                combined_texts.append(clean)
        if current_caps_words:
            combined_texts.append(' '.join(current_caps_words))
        
        for i, text in enumerate(combined_texts):
            # Clean up punctuation attached to words if necessary, 
            # but existing regexes are strict so stripped punct might be needed.
            # For now, we trust the split.
            
            clean_text = text.strip('.,;:()[]"')
            
            if not street_number:
                street_number = parse_street_number(clean_text)
                # Check if next token is a single letter suffix (e.g., "17" followed by "a")
                if street_number and i + 1 < len(combined_texts):
                    next_token = combined_texts[i + 1].strip('.,;:()[]"')
                    if len(next_token) == 1 and next_token.isalpha():
                        street_number = street_number + next_token
            
            # Unit number regex handles "Apt 1" which might be two tokens "Apt" and "1".
            # The original logic iterated over "lines" or "detected text blocks".
            # GOT-OCR returns one big string. "Apt 1" would be split into "Apt", "1".
            # parse_unit_number("Apt") -> None. parse_unit_number("1") -> None.
            # This is a problem. The existing logic relied on OCR blocks like "Apt 1".
            # Solution: Run parse_unit_number on the FULL text first, or sliding windows.
            # Actually, `parse_unit_number` uses `re.search`, so it finds it anywhere in the string!
            # So passing the FULL `res` to `parse_unit_number` is better/smarter.
            
            # Use individual tokens for street number and name, but full text for unit?
            if not street_number:
                 street_number = parse_street_number(clean_text)
            
            # Use individual tokens for street parts
            name = parse_street_name(clean_text)
            if name:
                street_names.append(name)

        # Try to find unit number in the full text string
        if not unit_number:
            unit_number = parse_unit_number(res)
            
        if street_names:
            street_name = ' '.join(street_names)
        
        return {
            'image_path': image_path,
            'filename': os.path.basename(image_path),
            'street_number': street_number,
            'street_name': street_name,
            'unit_number': unit_number,
            'confidence': 1.0, # Placeholder
            'processing_time': round(processing_time, 3)
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
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
    import gc
    total = len(image_paths)
    for i, path in enumerate(image_paths, 1):
        if i % 10 == 0 or i == total:
            print(f"[{i}/{total}] Processing {path.name}...")
        results[path.name] = extract_numbers(str(path))
        gc.collect()
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    return results


def process_batch_parallel(image_dir: str, output_file: str = 'results.json', workers: int = 4) -> dict:
    """
    Process all images in a directory using multiprocessing.
    Note: For GPU models, multiprocessing might cause CUDA OOM errors. 
    It is often better to use single threaded batch processing or careful worker management.
    """
    image_paths = [str(p) for p in Path(image_dir).glob('*.jpg')]
    image_paths += [str(p) for p in Path(image_dir).glob('*.jpeg')]
    image_paths += [str(p) for p in Path(image_dir).glob('*.png')]
    
    # Simple Pool map might attempt to re-initialize CUDA in forked processes.
    # For robust GPU usage, we usually avoid multiprocessing.Pool with CUDA.
    # However, to simulate the interface, we'll try. 
    # Best practice for GOT-OCR/Transformers is often sequential batching or using torch.multiprocessing with spawn.
    # Given the complexity, we'll default to 1 worker (sequential) if workers != 1 or warn user.
    # We will just implement the pool, but user should be aware of VRAM limits.
    
    with Pool(workers) as pool:
        results_list = pool.map(process_single_image, image_paths)
    
    merged = {}
    for res in results_list:
        for k, v in res.items():
            merged[Path(k).name] = v
            
    with open(output_file, 'w') as f:
        json.dump(merged, f, indent=2)
        
    return merged


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description='POD OCR (GOT-OCR2.0)')
    parser.add_argument('input', help='Path to image file or directory')
    parser.add_argument('-o', '--output', default='results.json', help='Output JSON file')
    parser.add_argument('-w', '--workers', type=int, default=1, help='Workers (Default 1 for GPU safety)')
    parser.add_argument('-c', '--confidence', type=float, default=0.95, help='Unused for GOT-OCR')
    
    args = parser.parse_args()
    input_path = Path(args.input)
    
    try:
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
    finally:
        # Always cleanup to free memory
        cleanup_ocr()


if __name__ == '__main__':
    main()
