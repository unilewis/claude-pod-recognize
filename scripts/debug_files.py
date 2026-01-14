import glob
import os
from pathlib import Path

print("--- GLB ---")
# g = glob.glob("images/*")
# print(f"Glob found {len(g)}: {sorted(g)}")

print("--- PATHLIB ---")
p = Path("images")
files = list(p.glob("*"))
print(f"Pathlib found {len(files)}: {sorted([f.name for f in files])}")

print("--- EXTENSIONS ---")
for f in files:
    print(f"{f.name} -> {f.suffix} (lower: {f.suffix.lower()})")
