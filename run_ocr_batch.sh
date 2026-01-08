#!/bin/bash
# run_ocr_batch.sh - Process POD images one by one and log results

# Configuration
IMAGE_DIR="${1:-./images}"
LOG_FILE="${2:-ocr_results.log}"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

echo "========================================" | tee -a "$LOG_FILE"
echo "OCR Batch Run: $TIMESTAMP" | tee -a "$LOG_FILE"
echo "Image Directory: $IMAGE_DIR" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Find all image files
find "$IMAGE_DIR" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | while read -r image; do
    filename=$(basename "$image")
    echo "" | tee -a "$LOG_FILE"
    echo "Processing: $filename" | tee -a "$LOG_FILE"
    echo "---" | tee -a "$LOG_FILE"
    
    # Run OCR and capture output
    result=$(python3 paddleocr/pod_ocr.py "$image" 2>&1)
    
    # Log the result
    echo "$result" | tee -a "$LOG_FILE"
    
    echo "---" | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Batch Complete: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "Results saved to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
