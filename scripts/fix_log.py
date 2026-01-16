import re
import json
import os

log_file = "logs/classification_benchmark_20260114_151846_results.log"
output_file = log_file + ".fixed"

# Use the robust parsing logic from the report generator
def parse_log(file_path):
    results = []
    prefix_pattern = re.compile(r"^\[(.*?)\] (.*?): (.*)")
    
    current_block = ""
    brace_count = 0
    in_block = False
    
    with open(file_path, 'r') as f:
        for line in f:
            match = prefix_pattern.match(line)
            if match:
                if in_block and current_block:
                    try_parse(current_block, results)
                    current_block = ""
                    brace_count = 0
                
                json_part = match.group(3)
                current_block = json_part
                brace_count = json_part.count('{') - json_part.count('}')
                
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

    if in_block and current_block:
        try_parse(current_block, results)
        
    return results

def try_parse(json_str, results):
    try:
        data = json.loads(json_str)
        results.append(data)
    except Exception as e:
        print(f"Skipping malformed block: {e}")

data = parse_log(log_file)
print(f"Parsed {len(data)} entries.")

# Filter out qwen and deepseek
keep_models = ["ministral-3:8b", "llama3.2-vision:11b", "blaifa/InternVL3_5:8b", "gemma3:4b", "devstral-small-2:24b", "gemma3:12b"]
filtered_data = [d for d in data if d.get('model') in keep_models]

print(f"Kept {len(filtered_data)} entries (removed qwen/deepseek).")

# Write back in the format expected by resume logic
with open(log_file, "w") as f:
    for entry in filtered_data:
        model = entry.get('model')
        filename = entry.get('filename')
        # Pretty print
        f.write(f"[{model}] {filename}: {json.dumps(entry, indent=2)}\n")

print(f"Overwrote {log_file} with cleaned data.")
