import os
import re
from pathlib import Path

LOG_DIR = Path("logs")
patterns = [
    # benchmark_results_20260108_205556.log -> benchmark_20260108_205556_results.log
    (r"benchmark_results_(\d{8}_\d{6})\.log", r"benchmark_\1_results.log"),
    
    # benchmark_analysis_20260108_205556.md -> benchmark_20260108_205556_analysis.md
    (r"benchmark_analysis_(\d{8}_\d{6})\.md", r"benchmark_\1_analysis.md"),
    
    # benchmark_live_20260108_212612.log -> benchmark_20260108_212612_live.log
    (r"benchmark_live_(\d{8}_\d{6})\.log", r"benchmark_\1_live.log"),
    
    # llm_benchmark_results_20260112_105141.log -> llm_benchmark_20260112_105141_results.log
    (r"llm_benchmark_results_(\d{8}_\d{6})\.log", r"llm_benchmark_\1_results.log"),
    
    # benchmark_debug_20260112_105141.log -> benchmark_20260112_105141_debug.log
    (r"benchmark_debug_(\d{8}_\d{6})\.log", r"benchmark_\1_debug.log"),
]

for filename in os.listdir(LOG_DIR):
    for pattern, repl in patterns:
        if match := re.match(pattern, filename):
            new_name = re.sub(pattern, repl, filename)
            old_path = LOG_DIR / filename
            new_path = LOG_DIR / new_name
            print(f"Renaming {filename} -> {new_name}")
            os.rename(old_path, new_path)
            break
