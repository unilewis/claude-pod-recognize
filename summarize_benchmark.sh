#!/bin/bash
# Extract and summarize benchmark results from log file

LOG_FILE="logs/benchmark_results_20260108_212612.log"
OUTPUT_FILE="logs/benchmark_analysis_$(date '+%Y%m%d_%H%M%S').md"

echo "Parsing $LOG_FILE..."

# Extract summary statistics
echo "# OCR Benchmark Results - images2 Folder (399 Images)" > "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "**Log File**: \`$(basename $LOG_FILE)\`" >> "$OUTPUT_FILE"
echo "**Benchmark Date**: $(date '+%Y-%m-%d %H:%M:%S')" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "---" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Count detections
echo "## Detection Summary" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "| Engine | Images Processed | Street Numbers Found |" >> "$OUTPUT_FILE"
echo "|--------|------------------|----------------------|" >> "$OUTPUT_FILE"

for engine in PaddleOCR EasyOCR GOT-OCR; do
    total=$(grep -c "\[$engine\]" "$LOG_FILE")
    detected=$(grep "\[$engine\]"  "$LOG_FILE" -A7 | grep '"street_number":' | grep -v 'null' | wc -l | tr -d ' ')
    echo "| $engine | $total | $detected |" >> "$OUTPUT_FILE"
done

echo "" >> "$OUTPUT_FILE"

# Performance stats
echo "## Performance Stats" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "| Engine | Avg Time | Min Time | Max Time |" >> "$OUTPUT_FILE"
echo "|--------|----------|----------|----------|" >> "$OUTPUT_FILE"

for engine in PaddleOCR EasyOCR GOT-OCR; do
    # Extract all processing times for this engine
    times=$(grep "\[$engine\]" "$LOG_FILE" -A7 | grep "processing_time" | awk -F': ' '{print $2}' | tr -d ',')
    
    if [ ! -z "$times" ]; then
        avg=$(echo "$times" | awk '{sum+=$1; count++} END {printf "%.2f", sum/count}')
        min=$(echo "$times" | sort -n | head -1)
        max=$(echo "$times" | sort -n | tail -1)
        echo "| $engine | ${avg}s | ${min}s | ${max}s |" >> "$OUTPUT_FILE"
    fi
done

echo "" >> "$OUTPUT_FILE"
echo "---" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "## Sample Results (First 20 Images)" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "| Filename | Engine | Street Number | Street Name | Time (s) |" >> "$OUTPUT_FILE"
echo "|----------|--------|---------------|-------------|----------|" >> "$OUTPUT_FILE"

# Extract first 20 images
count=0
for img_num in $(seq -f "%03g" 1 20); do
    filename="pod_${img_num}.jpg"
    
    for engine in PaddleOCR EasyOCR GOT-OCR; do
        # Extract data for this image and engine
        block=$(grep "\[$engine\] $filename:" "$LOG_FILE" -A7)
        
        if [ ! -z "$block" ]; then
            street_num=$(echo "$block" | grep '"street_number":' | sed 's/.*: "\?\([^",]*\).*/\1/')
            street_name=$(echo "$block" | grep '"street_name":' | sed 's/.*: "\?\([^",]*\).*/\1/' | cut -c1-25)
            proc_time=$(echo "$block" | grep '"processing_time":' | sed 's/.*: \([0-9.]*\).*/\1/')
            
            # Clean up null values
            [ "$street_num" = "null" ] && street_num="-"
            [ "$street_name" = "null" ] && street_name="-"
            
            # First row has filename
            if [ "$engine" = "PaddleOCR" ]; then
                echo "| $filename | $engine | $street_num | $street_name | $proc_time |" >> "$OUTPUT_FILE"
            else
                echo "| | $engine | $street_num | $street_name | $proc_time |" >> "$OUTPUT_FILE"
            fi
        fi
    done
done

echo "" >> "$OUTPUT_FILE"
echo "---" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "## Recommendations" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "- **Speed-Critical**: EasyOCR (fastest processing)" >> "$OUTPUT_FILE"
echo "- **Balanced**: PaddleOCR PP-OCRv4 (recommended for production)" >> "$OUTPUT_FILE"
echo "- **Maximum Recall**: GOT-OCR (highest detection rate)" >> "$OUTPUT_FILE"
echo "- **Best Accuracy**: PaddleOCR + GOT-OCR ensemble with voting" >> "$OUTPUT_FILE"

echo ""
echo "Analysis saved to: $OUTPUT_FILE"
echo ""
echo "Summary:"
for engine in PaddleOCR EasyOCR GOT-OCR; do
    total=$(grep -c "\[$engine\]" "$LOG_FILE")
    detected=$(grep "\[$engine\]" "$LOG_FILE" -A7 | grep '"street_number":' | grep -v 'null' | wc -l | tr -d ' ')
    rate=$(echo "scale=1; $detected * 100 / $total" | bc)
    echo "  $engine: $detected/$total ($rate%) detected"
done
