#!/bin/bash
# benchmark_ocrs.sh - Run all OCR engines against images and present results in a table

# Configuration
IMAGE_DIR="${1:-./images}"
LOG_FILE="benchmark_results.log"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Header for the table
printf "%-12s | %-12s | %-25s | %-20s | %-15s | %-10s | %-10s\n" "Engine" "Filename" "Street Number" "Street Name" "Unit Number" "Conf" "Time(s)"
printf "%s\n" "----------------------------------------------------------------------------------------------------------------------------------"

# Initialize Log
echo "Benchmark Run: $TIMESTAMP" > "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

# Define OCR engines (Name:Path)
# Note: Ensure these paths exist. 
ENGINES=(
    "PaddleOCR:paddleocr/pod_ocr.py"
    "EasyOCR:easyocr/pod_ocr.py"
    "GOT-OCR:got-ocr/pod_ocr.py"
    "TrOCR:trocr/pod_ocr.py"
)

# Determine Python Interpreter
if [ -f ".venv/bin/python" ]; then
    PYTHON_CMD=".venv/bin/python"
else
    PYTHON_CMD="python3"
fi
echo "Using Python: $PYTHON_CMD" >> "$LOG_FILE"

# Find image files
IMAGES=$(find "$IMAGE_DIR" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \))

if [ -z "$IMAGES" ]; then
    echo "No images found in $IMAGE_DIR"
    exit 1
fi

for engine_entry in "${ENGINES[@]}"; do
    IFS=":" read -r engine_name engine_script <<< "$engine_entry"
    
    # Check if script exists
    if [ ! -f "$engine_script" ]; then
        echo "Warning: Script $engine_script not found. Skipping." >> "$LOG_FILE"
        continue
    fi
    
    echo "Running $engine_name..." >> "$LOG_FILE"
    
    for image in $IMAGES; do
        # Extract filename for logging
        filename=$(basename "$image")
        
        # Run OCR script
        # We assume the script outputs JSON. We grab the last line just in case of other stdout noise
        # But our scripts print clean JSON.
        
        # Run and capture both stdout and stderr (but separate them if possible)
        # Using a temporary file for the json output to ensure we parse it correctly
        json_output=$($PYTHON_CMD "$engine_script" "$image" 2>/dev/null)
        
        # Log raw output
        echo "[$engine_name] $filename: $json_output" >> "$LOG_FILE"
        
        # Parse JSON with jq
        # Structure is usually { "filename.jpg": { ...data... } }
        # We want to extract the inner object.
        
        # We need to handle the case where the key is dynamic (filename)
        # jq 'to_entries[0].value' gets the first value regardless of key
        
        street_num=$(echo "$json_output" | jq -r 'to_entries[0].value.street_number // "null"')
        street_name=$(echo "$json_output" | jq -r 'to_entries[0].value.street_name // "null"')
        unit_num=$(echo "$json_output" | jq -r 'to_entries[0].value.unit_number // "null"')
        conf=$(echo "$json_output" | jq -r 'to_entries[0].value.confidence // 0')
        proc_time=$(echo "$json_output" | jq -r 'to_entries[0].value.processing_time // 0')
        error=$(echo "$json_output" | jq -r 'to_entries[0].value.error // empty')
        
        # Truncate strings for table
        s_name_short=${street_name:0:20}
        
        if [ ! -z "$error" ]; then
             printf "%-12s | %-12s | %-25s | %-20s | %-15s | %-10s | %-10s\n" "$engine_name" "${filename:0:12}" "ERROR" "ERROR" "-" "0.0" "$proc_time"
        else
             printf "%-12s | %-12s | %-25s | %-20s | %-15s | %-10s | %-10s\n" "$engine_name" "${filename:0:12}" "$street_num" "$s_name_short" "$unit_num" "$conf" "$proc_time"
        fi
        
    done
    printf "%s\n" "----------------------------------------------------------------------------------------------------------------------------------"
done

echo ""
echo "Full log saved to $LOG_FILE"
