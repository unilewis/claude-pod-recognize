#!/bin/bash
# Run LLM OCR tests on images folder using llm_ocr.py

# Default models
# Default models
MODELS=(
    "ministral-3:8b" 
    "llama3.2-vision:11b" 
    "blaifa/InternVL3_5:8b" 
    "qwen3-vl:8b" 
    "gemma3:4b"
    "devstral-small-2:24b"
    "deepseek-ocr:latest"
)
IMAGES_DIR=${IMAGES_DIR:-"images"}
OLLAMA_HOST=${OLLAMA_HOST:-"http://192.168.86.193:11434"}

# Check if a specific model was passed as argument
if [ ! -z "$1" ]; then
    MODELS=("$1")
fi

# Check if host was passed as second argument
if [ ! -z "$2" ]; then
    OLLAMA_HOST="$2"
fi

echo "========================================"
echo "Running LLM OCR Tests"
echo "Models: ${MODELS[@]}"
echo "Host: $OLLAMA_HOST"
echo "========================================"

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="logs/llm_benchmark_${TIMESTAMP}_results.log"
ANALYSIS_FILE="logs/llm_benchmark_${TIMESTAMP}_analysis.md"

echo "========================================"
echo "Starting Full LLM Benchmark Suite"
echo "Log File: $LOG_FILE"
echo "Models: ${MODELS[@]}"
echo "========================================"

for model in "${MODELS[@]}"; do
    echo ""
    echo ">>> Benchmarking Model: $model"
    echo "----------------------------------------"
    
    # Run the robust python runner
    # We pass the same log file to append results from all models
    python3 llm/benchmark_runner.py --model "$model" --host "$OLLAMA_HOST" --output "$LOG_FILE" --images "$IMAGES_DIR"
done

echo ""
echo "========================================"
echo "Generating Analysis Report..."
echo "========================================"

python3 scripts/analyze_benchmark.py "$LOG_FILE" > "$ANALYSIS_FILE"

echo "Report saved to $ANALYSIS_FILE"
echo ""
cat "$ANALYSIS_FILE"
