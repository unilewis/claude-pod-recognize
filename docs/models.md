# Models

## Model Architecture

The system uses a multi-stage approach for OCR extraction, but can also leverage Multimodal Large Language Models (LLMs) for direct extraction.

### Standard Pipeline (PaddleOCR / GOT-OCR)

1. **Detection**: Identifying text regions in the image.
2. **Recognition**: Converting text regions to string data.
3. **Extraction**: Parsing strings into structured data (Address, Unit, etc.).

### LLM Pipeline (Ollama)

Experimental support has been added for using local LLMs to directly extract structured address data from images.

#### Supported Models

The following models are supported via [Ollama](https://ollama.com):

- `qwen3-vl:8b`: Default. Good balance of speed and visual reasoning.
- `blaifa/InternVL3`: Strong Chinese/English capabilities.
- `deepseek-ocr`: Specialized for text-heavy images.
- `blaifa/InternVL3_5:8b`: Advanced multimodal capabilities (replacing llama4).

#### Usage

Use the `scripts/llm_ocr.py` utility to run extraction:

```bash
# Install dependencies
pip install requests

# Ensure Ollama is running
ollama serve

# Pull your desired model
ollama pull qwen3-vl:8b

# Run extraction
python3 scripts/llm_ocr.py \
  --image images2/pod_001.jpg \
  --model qwen3-vl:8b
```

**Output Example**:

```json
{
  "street_number": "123",
  "street_name": "Main St",
  "unit_number": "Apt 4B"
}
```
