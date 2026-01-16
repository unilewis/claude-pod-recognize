#!/usr/bin/env python3
import glob
import subprocess
import json
import os
import sys
import argparse
from datetime import datetime

print("DEBUG: Script started", flush=True)

DEFAULT_IMAGES_DIR = "images"
CLASSIFY_SCRIPT = "llm/llm_classify.py"
HOST = "http://192.168.86.193:11434"
LOG_FILE = f"logs/classification_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
LOG_JSON_FILE = f"logs/classification_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}_results.log"


MODELS = [
    "ministral-3:8b",
    "qwen3-vl:8b",
    "gemma3:4b",
    "devstral-small-2:24b"
]

def main():
    parser = argparse.ArgumentParser(description="Run batch image classification.")
    parser.add_argument("--dir", default=DEFAULT_IMAGES_DIR, help="Directory containing images to classify")
    parser.add_argument("--model", help="Specific model to run (optional)")
    parser.add_argument("--resume_from", help="Path to existing JSON log file to resume from")
    args = parser.parse_args()
    
    images_dir = args.dir
    images = sorted(glob.glob(f"{images_dir}/*"))
    # Case insensitive extension check
    valid_images = [img for img in images if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not valid_images:
        print(f"No images found in {images_dir}")
        sys.exit(1)

    models_to_run = MODELS
    if args.model:
        models_to_run = [args.model]

    # Resume Logic
    processed_keys = set()
    current_log_file = LOG_JSON_FILE
    
    if args.resume_from:
        if os.path.exists(args.resume_from):
            print(f"Resuming from {args.resume_from}...")
            # We must use the exact same file path to append to it effectively
            current_log_file = args.resume_from
            
            # Robust resume: Scan for lines starting with "[model] filename:"
            # regardless of whether the JSON that follows is single-line or multi-line.
            import re
            log_prefix_pattern = re.compile(r"^\[(.*?)\] (.*?):")
            
            with open(current_log_file, "r") as f:
                for line in f:
                    match = log_prefix_pattern.match(line)
                    if match:
                        model_name = match.group(1)
                        file_name = match.group(2)
                        processed_keys.add((model_name, file_name))
                        
            print(f"Found {len(processed_keys)} already processed items.")
        else:
            print(f"Warning: Resume file {args.resume_from} not found. Starting fresh.")
        
    print(f"Running classification on {len(valid_images)} images using {len(models_to_run)} models...")
    print(f"Logging JSON results to {current_log_file}")
    
    all_results = []
    
    # NOTE: Parsing existing multi-line JSON results to populate 'all_results' for the
    # final Markdown report is complex without a streaming JSON parser. 
    # For now, if resuming, the final Markdown report might only contain NEWLY processed items.
    # The user can generate a full report later by analyzing the full log file separately.
    
    # Open log file for appending
    with open(current_log_file, "a") as log_f:
        for model in models_to_run:
            print(f"\n>>> Benchmarking Model: {model}")
            
            for i, img_path in enumerate(valid_images):
                filename = os.path.basename(img_path)
                
                # SKIP CHECK
                if (model, filename) in processed_keys:
                    print(f"[{i+1}/{len(valid_images)}] Skipping {filename} (Already done)")
                    continue
                
                print(f"[{i+1}/{len(valid_images)}] Processing {img_path}...")
                try:
                    # Capture output of llm_classify.py
                    cmd = ["python3", CLASSIFY_SCRIPT, "--image", img_path, "--model", model, "--host", HOST]
                    result_proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    
                    # Output is JSON
                    data = json.loads(result_proc.stdout)
                    data['model'] = model # Add model info
                    all_results.append(data)
                    
                    # Log to file immediately in pretty format
                    filename = os.path.basename(img_path)
                    log_entry = f"[{model}] {filename}: {json.dumps(data, indent=2)}\n"
                    log_f.write(log_entry)
                    log_f.flush()
                    
                except subprocess.CalledProcessError as e:
                    print(f"Error processing {img_path}: {e.stderr}")
                    error_data = {
                        "filename": os.path.basename(img_path),
                        "model": model,
                        "category": "ERROR",
                        "confidence": 0.0,
                        "processing_time": 0.0,
                        "full_text": ""
                    }
                    all_results.append(error_data)
                    log_f.write(f"[{model}] {os.path.basename(img_path)}: {json.dumps(error_data, indent=2)}\n")
                    log_f.flush()

                except json.JSONDecodeError:
                    print(f"Error decoding JSON for {img_path}")
                    error_data = {
                        "filename": os.path.basename(img_path),
                        "model": model,
                        "category": "JSON_ERROR",
                        "confidence": 0.0,
                        "processing_time": 0.0,
                        "full_text": ""
                    }
                    all_results.append(error_data)
                    log_f.write(f"[{model}] {os.path.basename(img_path)}: {json.dumps(error_data, indent=2)}\n")
                    log_f.flush()

    # Generate Report
    generate_markdown_report(all_results, images_dir)

def generate_markdown_report(results, images_dir):
    total_images = len(set(r['filename'] for r in results))
    total_models = len(set(r['model'] for r in results))
    
    report_lines = []
    report_lines.append(f"# Image Classification & OCR Benchmark Results ({images_dir})")
    report_lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**Total Images:** {total_images}")
    report_lines.append(f"**Models Tested:** {total_models}")
    report_lines.append("")
    
    # Detailed Table
    report_lines.append("| Filename | Model | Category | Confidence | Text Length | Time (s) |")
    report_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    
    # Sort by filename then model
    results.sort(key=lambda x: (x.get('filename'), x.get('model')))
    
    for res in results:
        cat = res.get('category', 'UNKNOWN')
        text_len = len(res.get('full_text', '') or "")
        time_val = res.get('processing_time', 0)
        
        report_lines.append(f"| {res.get('filename')} | {res.get('model')} | **{cat}** | {res.get('confidence', 0.0)} | {text_len} chars | {time_val} |")
        
    report_lines.append("")
    report_lines.append("## full_text content (Sample)")
    
    # Add a section for full text sample (first 5 images)
    seen_images = set()
    for res in results:
        if res.get('filename') not in seen_images and len(seen_images) < 5:
            seen_images.add(res.get('filename'))
            report_lines.append(f"### {res.get('filename')} ({res.get('model')})")
            report_lines.append("```text")
            report_lines.append(str(res.get('full_text', '')).strip())
            report_lines.append("```")

    report_content = "\n".join(report_lines)
    
    # Save to file
    with open(LOG_FILE, "w") as f:
        f.write(report_content)
        
    # Print summary to stdout
    print(f"\nReport generated at {LOG_FILE}")

if __name__ == "__main__":
    main()

