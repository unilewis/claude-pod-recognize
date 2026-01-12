#!/usr/bin/env python3
"""Parse benchmark log file and create comprehensive analysis."""

import json
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime

def parse_log_file(log_path):
    """Parse benchmark log file and extract results."""
    results = defaultdict(dict)
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Extract all JSON results - pattern: [EngineName] filename.jpg: { ... }
    pattern = r'\[(\w+)\] (pod_\d+\.jpg): (\{[^}]+\})'
    matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
    
    for engine, filename, json_str in matches:
        try:
            data = json.loads(json_str)
            if filename in data:
                results[filename][engine] = data[filename]
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse JSON for {engine} {filename}: {e}")
            continue
    
    return results

def create_markdown_table(results):
    """Create markdown table with Engine as a column."""
    lines = []
    
    # Header
    lines.append("| Filename | Engine | Street Number | Street Name | Confidence | Time (s) |")
    lines.append("|----------|--------|---------------|-------------|------------|----------|")
    
    # Sort filenames
    sorted_files = sorted(results.keys())
    
    for filename in sorted_files:
        engines_data = results[filename]
        
        for engine in ['PaddleOCR', 'EasyOCR', 'GOT-OCR']:
            if engine not in engines_data:
                continue
                
            data = engines_data[engine]
            street_num = data.get('street_number') or 'null'
            street_name = data.get('street_name') or 'null'
            
            # Truncate street name if too long
            if len(str(street_name)) > 30:
                street_name = str(street_name)[:27] + '...'
            
            conf = data.get('confidence', 0.0)
            time_s = data.get('processing_time', 0.0)
            
            # First row has filename, subsequent rows are blank
            fname_col = filename if engine == 'PaddleOCR' else ''
            
            lines.append(f"| {fname_col} | {engine} | {street_num} | {street_name} | {conf:.3f} | {time_s:.2f} |")
    
    return '\\n'.join(lines)

def calculate_stats(results):
    """Calculate performance statistics."""
    stats = {
        'PaddleOCR': {'times': [], 'detected': 0},
        'EasyOCR': {'times': [], 'detected': 0},
        'GOT-OCR': {'times': [], 'detected': 0}
    }
    
    for filename, engines_data in results.items():
        for engine, data in engines_data.items():
            time_val = data.get('processing_time', 0.0)
            stats[engine]['times'].append(time_val)
            
            if data.get('street_number'):
                stats[engine]['detected'] += 1
    
    return stats

def create_analysis(results, stats):
    """Create analysis section."""
    total_images = len(results)
    
    lines = []
    lines.append("\\n## Performance Summary\\n")
    
    # Speed table
    lines.append("### Processing Speed\\n")
    lines.append("| Engine | Avg Time | Min Time | Max Time | Total Time |")
    lines.append("|--------|----------|----------|----------|------------|")
    
    for engine in ['PaddleOCR', 'EasyOCR', 'GOT-OCR']:
        times = stats[engine]['times']
        if times:
            avg = sum(times) / len(times)
            min_t = min(times)
            max_t = max(times)
            total = sum(times)
            lines.append(f"| {engine} | {avg:.2f}s | {min_t:.2f}s | {max_t:.2f}s | {total:.1f}s |")
    
    # Detection table
    lines.append("\\n### Detection Accuracy\\n")
    lines.append("| Engine | Street Numbers Found | Detection Rate |")
    lines.append("|--------|---------------------|----------------|")
    
    for engine in ['PaddleOCR', 'EasyOCR', 'GOT-OCR']:
        detected = stats[engine]['detected']
        rate = (detected / total_images) * 100
        lines.append(f"| {engine} | {detected}/{total_images} | {rate:.1f}% |")
    
    return '\\n'.join(lines)

def main():
    # Parse log file
    log_file = Path('logs/benchmark_results_20260108_212612.log')
    print(f"Parsing {log_file}...")
    
    results = parse_log_file(log_file)
    print(f"Found {len(results)} images")
    
    # Calculate stats
    stats = calculate_stats(results)
    
    # Create markdown output
    output = []
    output.append(f"# OCR Benchmark Results - images2 Folder ({len(results)} Images)\\n")
    output.append(f"**Log File**: `{log_file.name}`")
    output.append(f"**Benchmark Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
    output.append("---\\n")
    output.append("## Complete Results - All Fields (Sorted by Filename)\\n")
    output.append(create_markdown_table(results))
    output.append(create_analysis(results, stats))
    output.append("\\n---\\n")
    output.append("## Recommendations\\n")
    output.append("**Speed-Critical**: EasyOCR (fastest)")
    output.append("**Balanced**: PaddleOCR PP-OCRv4 (recommended)")
    output.append("**Maximum Recall**: GOT-OCR (highest detection rate)")
    
    # Save to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = Path(f'logs/benchmark_analysis_{timestamp}.md')
    
    with open(output_file, 'w') as f:
        f.write('\\n'.join(output))
    
    print(f"\\nAnalysis saved to: {output_file}")
    print(f"Total images: {len(results)}")
    print(f"\\nDetection rates:")
    for engine in ['PaddleOCR', 'EasyOCR', 'GOT-OCR']:
        detected = stats[engine]['detected']
        rate = (detected / len(results)) * 100
        print(f"  {engine}: {detected}/{len(results)} ({rate:.1f}%)")

if __name__ == '__main__':
    main()
