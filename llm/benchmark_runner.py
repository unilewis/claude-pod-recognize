#!/usr/bin/env python3
"""
Robust benchmark runner for LLM OCR.
Iterates through the images folder and runs the llm_ocr logic for each.
"""
import os
import sys
import glob
import time
import json
from pathlib import Path

# Add current directory to path so we can import llm_ocr if needed
sys.path.append(os.getcwd())
# Or better, just import the logic if possible.
# Since llm_ocr is a script, let's just use subprocess for isolation, or modify this to call the function directly.
# Calling function directly is better.

import importlib.util
spec = importlib.util.spec_from_file_location("llm_ocr", "llm/llm_ocr.py")
llm_ocr = importlib.util.module_from_spec(spec)
sys.modules["llm_ocr"] = llm_ocr
spec.loader.exec_module(llm_ocr)

def run_benchmark(images_dir="images", model="qwen3-vl:8b", output_file=None, host="http://localhost:11434"):
    image_extensions = {".jpg", ".jpeg", ".png"}
    images = []
    
    # Use pathlib for case-insensitive matching if needed
    p = Path(images_dir)
    for file in p.glob("*"):
        if file.suffix.lower() in image_extensions:
            images.append(str(file))
        else:
            print(f"Skipping non-image file: {file}")
            
    images.sort()
    print(f"Found {len(images)} images in {images_dir}")
    
    results = []
    
    # Open file for append mode if specified
    f = None
    if output_file:
        f = open(output_file, "a")
        print(f"Logging to {output_file}")

    for i, img_path in enumerate(images):
        print(f"[{i+1}/{len(images)}] Processing {img_path}...")
        
        try:
            start = time.time()
            data = llm_ocr.extract_address(img_path, model, host)
            end = time.time()
            
            filename = Path(img_path).name
            full_result = {
                filename: {
                    "image_path": img_path,
                    "filename": filename,
                    "street_number": data.get("street_number"),
                    "street_name": data.get("street_name"),
                    "unit_number": data.get("unit_number"),
                    "confidence": 1.0,
                    "processing_time": round(end - start, 3)
                }
            }
            
            log_line = f"[{model}] {filename}: {json.dumps(full_result, indent=2)}"
            
            # Print to stdout
            print(log_line)
            
            # Print to log file
            if f:
                f.write(log_line + "\n")
                f.flush()
                
            # Sleep to be gentle on local CPU/Memory
            time.sleep(0.5)
            
        except BaseException as e:
            err_msg = f"ERROR processing {img_path}: {e}"
            print(err_msg, file=sys.stderr)
            if f:
                f.write(err_msg + "\n")

    if f:
        f.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="Log file path")
    parser.add_argument("--model", default="qwen3-vl:8b", help="Model name")
    parser.add_argument("--images", default="images", help="Directory containing images")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama API host URL")
    args = parser.parse_args()
    
    if args.output:
        run_benchmark(images_dir=args.images, output_file=args.output, model=args.model, host=args.host)
    else:
        # Generate default log file matching new convention
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_log = f"logs/llm_benchmark_{ts}_results.log"
        os.makedirs("logs", exist_ok=True)
        print(f"No output file specified. Using default: {default_log}")
        run_benchmark(images_dir=args.images, output_file=default_log, model=args.model, host=args.host)
