#!/usr/bin/env python3
"""Parse benchmark log file and generate table + analysis."""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

def parse_benchmark_log(log_path):
    """Parse benchmark log and extract all results."""
    results = []
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Pattern to match engine results like [PaddleOCR] pod_001.jpg: { ... }
    pattern = r'\[([^\]]+)\] (pod_?\d+\.(?:jpg|jpeg|png)): (\{[\s\S]*?\n\})'
    
    matches = re.findall(pattern, content)
    
    for engine, filename, json_str in matches:
        try:
            data = json.loads(json_str)
            # Get the inner data (keyed by filename)
            inner = data.get(filename, data)
            results.append({
                'engine': engine,
                'filename': filename,
                'image_path': inner.get('image_path', ''),
                'street_number': inner.get('street_number'),
                'street_name': inner.get('street_name'),
                'unit_number': inner.get('unit_number'),
                'confidence': inner.get('confidence', 0),
                'processing_time': inner.get('processing_time', 0)
            })
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse JSON for {engine} {filename}: {e}", file=sys.stderr)
    
    return results

def generate_grouped_table(results):
    """Generate table grouped by image with dividers."""
    # Group by filename
    by_filename = defaultdict(list)
    for r in results:
        by_filename[r['filename']].append(r)
    
    # Sort filenames naturally (pod_1.jpg, pod_2.jpg, ... pod_10.jpg)
    def natural_sort_key(s):
        import re
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]
    
    sorted_filenames = sorted(by_filename.keys(), key=natural_sort_key)
    
    # Define column widths
    col_widths = {
        'image': 14,
        'engine': 25,
        'street_num': 20,
        'street_name': 32,
        'unit': 8,
        'conf': 6,
        'time': 8
    }
    
    total_width = sum(col_widths.values()) + 6 * 3  # 6 separators with 3 chars each " | "
    
    # Header separator
    header_sep = "=" * total_width
    row_sep = "-" * total_width
    
    print(header_sep)
    print(f"{'Image':<{col_widths['image']}} | {'Engine':<{col_widths['engine']}} | {'Street Num':<{col_widths['street_num']}} | {'Street Name':<{col_widths['street_name']}} | {'Unit':<{col_widths['unit']}} | {'Conf':<{col_widths['conf']}} | {'Time(s)':<{col_widths['time']}}")
    print(header_sep)
    
    # Engine order preference
    engine_order = ['PaddleOCR', 'EasyOCR', 'GOT-OCR', 'TrOCR']
    
    for i, filename in enumerate(sorted_filenames):
        rows = by_filename[filename]
        
        # Sort rows by engine order
        def engine_sort_key(r):
            try:
                return engine_order.index(r['engine'])
            except ValueError:
                return len(engine_order)
        
        sorted_rows = sorted(rows, key=engine_sort_key)
        
        for j, r in enumerate(sorted_rows):
            # First row gets the filename, others get empty
            img_col = filename if j == 0 else ""
            
            street_num = r['street_number'] or '-'
            street_name = r['street_name'] or '-'
            # Truncate street name if too long
            if len(street_name) > col_widths['street_name'] - 3:
                street_name = street_name[:col_widths['street_name'] - 3] + "..."
            unit = r['unit_number'] or '-'
            conf = f"{r['confidence']:.3f}" if r['confidence'] else '0.000'
            # Shorten confidence display
            if conf.startswith('0.'):
                conf = conf[1:]  # Remove leading 0
            elif conf == '1.000':
                conf = '1.0'
            time_str = f"{r['processing_time']:.2f}"
            
            print(f"{img_col:<{col_widths['image']}} | {r['engine']:<{col_widths['engine']}} | {street_num:<{col_widths['street_num']}} | {street_name:<{col_widths['street_name']}} | {unit:<{col_widths['unit']}} | {conf:<{col_widths['conf']}} | {time_str:<{col_widths['time']}}")
        
        # Print separator between images (except for last one)
        if i < len(sorted_filenames) - 1:
            print(row_sep)

def generate_analysis(results):
    """Generate statistical analysis."""
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80 + "\n")
    
    # Group by engine
    by_engine = defaultdict(list)
    for r in results:
        by_engine[r['engine']].append(r)
    
    # Overall stats
    print("### Per-Engine Statistics\n")
    print(f"{'Engine':<12} | {'Images':>6} | {'Avg Conf':>8} | {'Avg Time':>8} | {'Min Time':>8} | {'Max Time':>8} | {'Total Time':>10}")
    print("-" * 80)
    
    for engine in sorted(by_engine.keys()):
        data = by_engine[engine]
        count = len(data)
        avg_conf = sum(r['confidence'] for r in data) / count if count else 0
        times = [r['processing_time'] for r in data]
        avg_time = sum(times) / count if count else 0
        min_time = min(times) if times else 0
        max_time = max(times) if times else 0
        total_time = sum(times)
        print(f"{engine:<12} | {count:>6} | {avg_conf:>8.3f} | {avg_time:>8.3f} | {min_time:>8.3f} | {max_time:>8.3f} | {total_time:>10.1f}s")
    
    # Detection rates
    print("\n### Detection Rates\n")
    print(f"{'Engine':<12} | {'Has Street #':>14} | {'Has Street Name':>16} | {'Has Unit #':>12} | {'Has Any Data':>14}")
    print("-" * 80)
    
    for engine in sorted(by_engine.keys()):
        data = by_engine[engine]
        count = len(data)
        has_num = sum(1 for r in data if r['street_number'])
        has_name = sum(1 for r in data if r['street_name'])
        has_unit = sum(1 for r in data if r['unit_number'])
        has_any = sum(1 for r in data if r['street_number'] or r['street_name'] or r['unit_number'])
        print(f"{engine:<12} | {has_num:>3}/{count} ({100*has_num/count:>4.1f}%) | {has_name:>4}/{count} ({100*has_name/count:>5.1f}%) | {has_unit:>2}/{count} ({100*has_unit/count:>4.1f}%) | {has_any:>3}/{count} ({100*has_any/count:>5.1f}%)")
    
    # Key findings
    print("\n### Key Findings\n")
    
    engines = list(by_engine.keys())
    
    # Fastest average
    fastest = min(engines, key=lambda e: sum(r['processing_time'] for r in by_engine[e]) / len(by_engine[e]))
    fastest_time = sum(r['processing_time'] for r in by_engine[fastest]) / len(by_engine[fastest])
    
    # Highest confidence
    best_conf = max(engines, key=lambda e: sum(r['confidence'] for r in by_engine[e]) / len(by_engine[e]))
    best_conf_val = sum(r['confidence'] for r in by_engine[best_conf]) / len(by_engine[best_conf])
    
    # Most data extracted
    best_detection = max(engines, key=lambda e: sum(1 for r in by_engine[e] if r['street_number'] or r['street_name']))
    detection_pct = sum(1 for r in by_engine[best_detection] if r['street_number'] or r['street_name']) / len(by_engine[best_detection]) * 100
    
    print(f"1. Fastest Engine: {fastest} (avg {fastest_time:.3f}s per image)")
    print(f"2. Highest Confidence: {best_conf} (avg {best_conf_val:.3f})")
    print(f"3. Best Detection Rate: {best_detection} ({detection_pct:.1f}% images with data)")
    
    # Total processing time
    print("\n### Total Processing Time\n")
    for engine in sorted(by_engine.keys()):
        total = sum(r['processing_time'] for r in by_engine[engine])
        mins = total / 60
        print(f"- {engine}: {total:.1f}s ({mins:.1f} minutes)")

def main():
    if len(sys.argv) < 2:
        log_path = "/Users/llewis/Code/claude-pod-recognize/logs/benchmark_20260108_212612_results.log"
    else:
        log_path = sys.argv[1]
    
    results = parse_benchmark_log(log_path)
    
    print("OCR BENCHMARK RESULTS")
    print(f"Log File: {Path(log_path).name}")
    print(f"Total Records: {len(results)}")
    print(f"Images: {len(set(r['filename'] for r in results))}")
    print(f"Engines: {', '.join(sorted(set(r['engine'] for r in results)))}")
    print()
    
    generate_grouped_table(results)
    generate_analysis(results)

if __name__ == "__main__":
    main()
