#!/usr/bin/env python3
"""
Combined LLM Benchmark Runner
Iterates through images and runs the combined classification & extraction logic.
"""
import os
import sys
import glob
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

# Import the combined logic
import importlib.util
spec = importlib.util.spec_from_file_location("llm_combined", "llm/llm_combined.py")
llm_combined = importlib.util.module_from_spec(spec)
sys.modules["llm_combined"] = llm_combined
spec.loader.exec_module(llm_combined)

DEFAULT_MODELS = [
    "qwen3-vl:8b",
    "gemma3:4b",
    "ministral-3:8b",
    "llama3.2-vision:11b"
]

def run_benchmark(images_dir="images", models=None, output_file=None, host="http://localhost:11434"):
    if models is None:
        models = DEFAULT_MODELS
        
    image_extensions = {".jpg", ".jpeg", ".png"}
    images = []
    
    p = Path(images_dir)
    if not p.exists():
        print(f"Error: Directory {images_dir} does not exist.")
        sys.exit(1)
        
    for file in p.glob("*"):
        if file.suffix.lower() in image_extensions:
            images.append(str(file))
            
    images.sort()
    print(f"Target Images: {len(images)} in {images_dir}")
    print(f"Target Models: {models}")
    
    # Resume Logic
    processed_keys = set()
    if output_file and os.path.exists(output_file):
        print(f"Resuming from {output_file}...")
        try:
            with open(output_file, 'r') as f:
                for line in f:
                    # Expecting format: [model] filename: JSON
                    if line.startswith("["):
                        parts = line.split("] ", 1)
                        if len(parts) > 1:
                            model_name = parts[0][1:]
                            meta_part = parts[1]
                            fname = meta_part.split(":", 1)[0].strip()
                            processed_keys.add((model_name, fname))
            print(f"Found {len(processed_keys)} already processed items.")
        except Exception as e:
            print(f"Warning: Could not parse resume file correctly: {e}")

    # Open file for append mode if specified
    f = None
    if output_file:
        f = open(output_file, "a")
        print(f"Logging to {output_file}")

    all_results = []
    
    # Iterate through models
    for model in models:
        print(f"\n>>> Benchmarking Model: {model}")
        
        for i, img_path in enumerate(images):
            filename = Path(img_path).name
            
            if (model, filename) in processed_keys:
                # print(f"Skipping {filename} (Already done)")
                continue
                
            print(f"[{i+1}/{len(images)}] Processing {img_path}...")
            
            try:
                start = time.time()
                data = llm_combined.process_image(img_path, model, host)
                end = time.time()
                
                # Ensure critical fields exist
                data["filename"] = filename
                data["processing_time"] = round(end - start, 3)
                data["model"] = model
                
                log_line = f"[{model}] {filename}: {json.dumps(data, indent=2)}"
                
                # Print to stdout (concise)
                cat = data.get('category', 'unknown')
                print(f"  -> {cat}, {data.get('street_name') or '-'}, Time: {data.get('processing_time')}s")
                
                # Print to log file (full)
                if f:
                    f.write(log_line + "\n")
                    f.flush()
                
                # Sleep to be gentle on local resources
                time.sleep(0.5)
                
            except KeyboardInterrupt:
                print("\nBenchmark interrupted by user.")
                if f: f.close()
                return
            except Exception as e:
                err_msg = f"ERROR processing {img_path}: {e}"
                print(err_msg, file=sys.stderr)
                if f:
                    f.write(f"[{model}] {filename}: {{ \"error\": \"{str(e)}\" }}\n")

    if f:
        f.close()
        
    # Generate Analysis Report
    if output_file:
        generate_report(output_file)

def generate_report(log_file):
    print(f"\nGenerating analysis report for {log_file}...")
    
    import re
    
    results = []
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            
        # Regex to match: [Model] filename: { ... }
        # Matches newline blocks for the JSON
        pattern = r'\[([^\]]+)\] ([^:]+): (\{[\s\S]*?\n\})'
        
        matches = re.findall(pattern, content)
        
        for model, filename, json_str in matches:
            try:
                data = json.loads(json_str)
                # Ensure model is set if missing in JSON (though it should be there)
                if "model" not in data:
                    data["model"] = model
                if "error" not in data:
                    results.append(data)
            except:
                pass
                
    except Exception as e:
        print(f"Error reading log for report: {e}")
        return

    if not results:
        print("No results found to analyze.")
        return

    # Create MD file
    report_file = log_file.replace(".log", "_analysis.md")
    
    lines = []
    lines.append(f"# Benchmark Analysis Report")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Log File:** `{os.path.basename(log_file)}`")
    lines.append(f"**Total Records:** {len(results)}")
    lines.append("")
    
    # 1. Classification Summary
    lines.append("## 1. Classification Summary")
    lines.append("| Model | Shipping Label | Address | Map | Unknown |")
    lines.append("| :--- | :--- | :--- | :--- | :--- |")
    
    models = sorted(list(set(r['model'] for r in results)))
    
    for m in models:
        m_res = [r for r in results if r['model'] == m]
        count = len(m_res)
        if count == 0: continue
        
        shop = sum(1 for r in m_res if "shipping" in r.get('category', ''))
        addr = sum(1 for r in m_res if "address" in r.get('category', ''))
        game_map = sum(1 for r in m_res if "map" in r.get('category', ''))
        unk = count - shop - addr - game_map
        
        lines.append(f"| {m} | {shop} | {addr} | {game_map} | {unk} |")
        
    lines.append("")
        
    # 2. Extraction Performance
    lines.append("## 2. Address Extraction Performance")
    lines.append("| Model | Avg Time (s) | Has Street # | Has Street Name | Has Unit |")
    lines.append("| :--- | :--- | :--- | :--- | :--- |")
    
    for m in models:
        m_res = [r for r in results if r['model'] == m]
        count = len(m_res)
        if count == 0: continue
        
        avg_time = sum(r.get('processing_time', 0) for r in m_res) / count
        has_num = sum(1 for r in m_res if r.get('street_number'))
        has_name = sum(1 for r in m_res if r.get('street_name'))
        has_unit = sum(1 for r in m_res if r.get('unit_number'))
        
        lines.append(f"| {m} | {avg_time:.2f} | {has_num} ({100*has_num/count:.0f}%) | {has_name} ({100*has_name/count:.0f}%) | {has_unit} ({100*has_unit/count:.0f}%) |")

    lines.append("")
    
    # 3. Detailed Results
    lines.append("## 3. Detailed Results")
    lines.append("| Filename | Model | Category | Street # | Street Name | Unit | Time (s) |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    
    # Sort by filename
    results.sort(key=lambda x: x.get('filename', ''))
    
    for r in results:
        fname = r.get('filename', '')
        model = r.get('model', '')
        cat = r.get('category', '')
        s_num = r.get('street_number') or '-'
        s_name = r.get('street_name') or '-'
        u_num = r.get('unit_number') or '-'
        time_val = r.get('processing_time', 0)
        
        lines.append(f"| {fname} | {model} | {cat} | {s_num} | {s_name} | {u_num} | {time_val} |")

    lines.append("")
    
    with open(report_file, "w") as f:
        f.write("\n".join(lines))
        
    print(f"Report saved to {report_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="Log file path")
    parser.add_argument("--models", nargs='+', help="List of models to run")
    parser.add_argument("--images", default="images", help="Directory containing images")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama API host URL")
    args = parser.parse_args()
    
    output_path = args.output
    if not output_path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"logs/combined_benchmark_{ts}_results.log"
        os.makedirs("logs", exist_ok=True)
        print(f"No output file specified. Using default: {output_path}")

    run_benchmark(images_dir=args.images, models=args.models, output_file=output_path, host=args.host)
