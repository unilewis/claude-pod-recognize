#!/usr/bin/env python3
import json
import sys
import os
import re
from datetime import datetime

# Import the reporting function/logic or just re-implement it briefly for standalone use
# Re-implementing is safer to avoid import side-effects from the batch runner
def generate_markdown_report(log_file, images_dir="images"):
    results = []
    
    # Robust parsing of the log file
    # Format: [model] filename: {json}
    # But now we know it's multiline JSON, so we can't just split lines easily if we treat it line-by-line purely.
    # However, the previous script wrote: f"[{model}] {filename}: {json.dumps(data, indent=2)}\n"
    # This means the log file is a mix of prefix and indented JSON. 
    # Actually, valid JSON objects are enclosed in braces. 
    # A simple strategy: Read the whole file, use a regex to find the JSON parts or the prefixes.
    
    # Simpler: The script `run_classification_batch.py` writes output line-by-line? 
    # No, I changed it to `indent=2`. This makes the log file technically invalid as a line-by-line format for simple iteration.
    # But wait, I added logic in `run_classification_batch.py` to parse it for RESUMING.
    # "Robust resume: Scan for lines starting with "[model] filename:"
    
    # The JSON data itself is spread across multiple lines. 
    # We need to extract the JSON blobs. 
    
    with open(log_file, 'r') as f:
        content = f.read()

    # Pattern: [model] filename: { ... }
    # We can rely on the prefix to start a block.
    # But the JSON spans multiple lines.
    # Let's try splitting by the prefix pattern.
    
    pattern = re.compile(r"^\[(.*?)\] (.*?): (\{.*)", re.MULTILINE)
    
    # This is tricky with multiline. 
    # Alternative: The log file structure is:
    # [model] file: {
    #   ...
    # }
    # [model] file: {
    #   ...
    # }
    
    # We can read line by line. If a line matches the prefix, we start capturing.
    # We count braces to find the end of the JSON object.
    
    prefix_pattern = re.compile(r"^\[(.*?)\] (.*?): (.*)")
    
    current_block = ""
    brace_count = 0
    in_block = False
    
    with open(log_file, 'r') as f:
        for line in f:
            match = prefix_pattern.match(line)
            if match:
                # If we were building a block, force finish it (shouldn't happen if logic is correct, but for safety)
                if in_block and current_block:
                    try_parse(current_block, results)
                    current_block = ""
                    brace_count = 0
                
                # Start new block
                json_part = match.group(3)
                current_block = json_part
                brace_count = json_part.count('{') - json_part.count('}')
                
                # If balanced immediately (single line), parse and reset
                if brace_count == 0 and current_block.strip():
                     try_parse(current_block, results)
                     current_block = ""
                     in_block = False
                else:
                    in_block = True
                continue
            
            if in_block:
                current_block += line
                brace_count += line.count('{') - line.count('}')
                
                if brace_count == 0:
                    try_parse(current_block, results)
                    current_block = ""
                    in_block = False

    # Check trailing
    if in_block and current_block:
        try_parse(current_block, results)

    # Now generate the MD
    output_md = log_file.replace("_results.log", "_analysis.md")
    
    # Filter out duplicates (keep latest)
    # Key: (model, filename)
    unique_results = {}
    for r in results:
        key = (r.get('model'), r.get('filename'))
        unique_results[key] = r
    
    results = list(unique_results.values())
    
    total_images = len(set(r.get('filename') for r in results))
    total_models = len(set(r.get('model') for r in results))
    
    report_lines = []
    report_lines.append(f"# Analysis Report")
    report_lines.append(f"**Source:** `{log_file}`")
    report_lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**Total Images:** {total_images}")
    report_lines.append(f"**Models Tested:** {total_models}")
    report_lines.append("")
    
    report_lines.append("| Filename | Model | Category | Confidence | Text Length | Time (s) |")
    report_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    
    # Sort
    results.sort(key=lambda x: (x.get('filename', ''), x.get('model', '')))
    
    for res in results:
        cat = res.get('category', 'UNKNOWN')
        text_len = len(res.get('full_text', '') or "")
        time_val = res.get('processing_time', 0)
        mdl = res.get('model', 'unknown')
        fname = res.get('filename', 'unknown')
        conf = res.get('confidence', 0.0)
        
        report_lines.append(f"| {fname} | {mdl} | **{cat}** | {conf} | {text_len} chars | {time_val} |")
        
    report_lines.append("")
    report_lines.append("## Full Text Samples (First 3 instances)")
    count = 0
    for res in results:
        if count < 3 and res.get('full_text'):
            count += 1
            report_lines.append(f"### {res.get('filename')} ({res.get('model')})")
            report_lines.append("```text")
            report_lines.append(str(res.get('full_text', '')).strip())
            report_lines.append("```")

    with open(output_md, "w") as f:
        f.write("\n".join(report_lines))
        
    print(f"Analysis generated at: {output_md}")

def try_parse(lines, results):
    try:
        json_str = "".join(lines)
        data = json.loads(json_str)
        results.append(data)
    except Exception as e:
        print(f"Failed to parse block: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 generate_report.py <log_file>")
        sys.exit(1)
    generate_markdown_report(sys.argv[1])
