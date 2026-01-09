#!/bin/bash
# benchmark_ocrs.sh - Run all OCR engines against images and present results in a table

# Configuration
IMAGE_DIR="${1:-./images}"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/benchmark_results_$TIMESTAMP.log"

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Header for the table
printf "%-12s | %-12s | %-25s | %-20s | %-15s | %-10s | %-10s\n" "Engine" "Filename" "Street Number" "Street Name" "Unit Number" "Conf" "Time(s)"
printf "%s\n" "----------------------------------------------------------------------------------------------------------------------------------"

# Initialize Log
echo "Benchmark Run: $(date '+%Y-%m-%d %H:%M:%S')" > "$LOG_FILE"
echo "Image Directory: $IMAGE_DIR" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

# Define OCR engines (Name:Path)
# Note: Ensure these paths exist. 
ENGINES=(
    "PaddleOCR:paddleocr/pod_ocr.py"
    "EasyOCR:easyocr/pod_ocr.py"
    "GOT-OCR:got-ocr/pod_ocr.py"
    # "TrOCR:trocr/pod_ocr.py"
)

# Determine Python Interpreter
if [ -f ".venv/bin/python" ]; then
    PYTHON_CMD=".venv/bin/python"
else
    PYTHON_CMD="python3"
fi
echo "Using Python: $PYTHON_CMD" >> "$LOG_FILE"

# Find image files
IMAGE_LIST=$(find "$IMAGE_DIR" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | sort -V)
total_images=$(echo "$IMAGE_LIST" | wc -l | xargs)

if [ -z "$IMAGE_LIST" ]; then
    echo "No images found in $IMAGE_DIR"
    exit 1
fi

echo "Processing $total_images images sequentially..." | tee -a "$LOG_FILE"

for engine_entry in "${ENGINES[@]}"; do
    # Skip commented out engines
    [[ "$engine_entry" =~ ^[[:space:]]*# ]] && continue
    
    IFS=":" read -r engine_name engine_script <<< "$engine_entry"
    
    # Check if script exists
    if [ ! -f "$engine_script" ]; then
        echo "Warning: Script $engine_script not found. Skipping." | tee -a "$LOG_FILE"
        continue
    fi
    
    echo "========================================" | tee -a "$LOG_FILE"
    echo "Running $engine_name..." | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    
    count=0
    for image in $IMAGE_LIST; do
        ((count++))
        filename=$(basename "$image")
        
        # Limit processing if IMAGES_LIMIT is set
        if [ ! -z "$IMAGES_LIMIT" ] && [ "$count" -gt "$IMAGES_LIMIT" ]; then
            echo "Limit reached ($IMAGES_LIMIT images). Moving to next engine." | tee -a "$LOG_FILE"
            break
        fi
        
        # Run OCR script in a fresh process
        # This ensures the OS reclaims memory when the process exits
        json_output=$($PYTHON_CMD "$engine_script" "$image" 2>/dev/null)
        
        # Log raw output
        echo "[$engine_name] $filename: $json_output" >> "$LOG_FILE"
        
        # Check if output is empty
        if [ -z "$json_output" ]; then
            printf "[%-7s] %-12s | %-12s | %-10s | %-10s\n" "$count/$total_images" "$engine_name" "${filename:0:12}" "EMPTY OUTPUT" "0.0s"
            continue
        fi

        # Parse JSON
        street_num=$(echo "$json_output" | jq -r 'to_entries[0].value.street_number // "null"')
        street_name=$(echo "$json_output" | jq -r 'to_entries[0].value.street_name // "null"')
        unit_num=$(echo "$json_output" | jq -r 'to_entries[0].value.unit_number // "null"')
        conf=$(echo "$json_output" | jq -r 'to_entries[0].value.confidence // 0')
        proc_time=$(echo "$json_output" | jq -r 'to_entries[0].value.processing_time // 0')
        error=$(echo "$json_output" | jq -r 'to_entries[0].value.error // empty')
        
        s_name_short=${street_name:0:20}
        
        if [ ! -z "$error" ]; then
             printf "[%-7s] %-12s | %-12s | %-12s | %-10s | %-10s\n" "$count/$total_images" "$engine_name" "${filename:0:12}" "ERROR" "0.0" "$proc_time"
        else
             printf "[%-7s] %-12s | %-12s | %-12s | %-20s | %-6s | %-7s\n" "$count/$total_images" "$engine_name" "${filename:0:12}" "$street_num" "$s_name_short" "$conf" "$proc_time"
        fi
        
        # Give the OS a moment to reclaim memory
        sleep 0.1
    done
    printf "%s\n" "----------------------------------------------------------------------------------------------------------------------------------"
done

echo ""
echo "Full log saved to $LOG_FILE"
