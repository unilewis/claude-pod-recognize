#!/usr/bin/env python3
"""
LLM OCR Tool via Ollama

Extracts structured address data (Street Number, Street Name, Unit Number) from images
using local Multimodal LLMs via the Ollama API.

Supported Models (must be pulled locally via `ollama pull <model>`):
- qwen3-vl:8b (Default)
- blaifa/InternVL3
- deepseek-ocr
- llama4:scout
"""

import argparse
import base64
import json
import json.decoder
import sys
import requests
from pathlib import Path

# Mapping for any specific model quirks if needed, currently generic.
SUPPORTED_MODELS = [
    "qwen3-vl:8b",
    "blaifa/InternVL3", 
    "deepseek-ocr",
    "blaifa/InternVL3_5:8b"
]

def encode_image(image_path):
    """Encodes an image to base64."""
    if not Path(image_path).exists():
        print(f"Error: Image not found at {image_path}", file=sys.stderr)
        sys.exit(1)
        
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_address(image_path, model, host="http://localhost:11434"):
    """Sends image to Ollama for structured extraction."""
    
    base64_image = encode_image(image_path)
    
    # Structured extraction prompt
    # Structured extraction prompt
    prompt = """
    Analyze this image and extract the delivery address details.
    Return ONLY a valid JSON object with the following keys:
    - street_number (string or null)
    - street_name (string or null)
    - unit_number (string or null)

    If a field is not visible, use null. do not include markdown formatting like ```json.
    """

    # DeepSeek-OCR specific prompt
    if "deepseek-ocr" in model:
        prompt = "<|grounding|>Extract the delivery address details."
        
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [base64_image],
        "stream": False,
    }
    
    # Only force JSON format for models that support it reliably
    if "deepseek-ocr" not in model:
        payload["format"] = "json"

    try:
        response = requests.post(f"{host}/api/generate", json=payload)
        response.raise_for_status()
        
        result = response.json()
        raw_response = result.get("response", "")
        
        # DeepSeek-OCR specific parsing (returns <|ref|>...<|/det|> content)
        if "deepseek-ocr" in model:
            import re
            # Remove tags like <|ref|>...<|/det|>
            cleaned_text = re.sub(r'<\|.*?\|>', '', raw_response)
            # Remove bounding box coords like [[123, 456, ...]]
            cleaned_text = re.sub(r'\[\[.*?\]\]', '', cleaned_text)
            # Clean up extra whitespace lines
            lines = [line.strip() for line in cleaned_text.splitlines() if line.strip()]
            full_text = " ".join(lines)
            
            # Return as simple structure since we can't easily parse fields without LLM post-processing
            return {
                "street_number": None,
                "street_name": full_text[:100] + "..." if len(full_text) > 100 else full_text, # Truncate for display
                "unit_number": None, 
                "full_text": full_text
            }
        
        # Fallback for models that put output in 'thinking' (like Qwen sometimes)
        if not raw_response and result.get("thinking"):
            raw_response = result.get("thinking")
        
        # Clean up potential markdown code blocks if the model ignores the "format: json" sometimes
        cleaned_response = raw_response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
            
        return json.loads(cleaned_response)

    except requests.exceptions.ConnectionError:
        raise ConnectionError(f"Could not connect to Ollama at {host}. Is it running?")
    except json.JSONDecodeError:
        raise ValueError(f"Model did not return valid JSON. Full API Result: {result}")
    except Exception as e:
        raise RuntimeError(f"Extraction failed: {e}")

import time

def main():
    parser = argparse.ArgumentParser(description="Extract address from image using local LLMs via Ollama.")
    parser.add_argument("--image", required=True, help="Path to the image file")
    parser.add_argument("--model", default="qwen3-vl:8b", help=f"Ollama model to use. Options: {', '.join(SUPPORTED_MODELS)}")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama API host")
    parser.add_argument("--benchmark", action="store_true", help="Output in benchmark log format")
    
    args = parser.parse_args()
    
    try:
        if not args.benchmark:
            print(f"Processing {args.image} with {args.model}...")
            
        start_time = time.time()
        data = extract_address(args.image, args.model, args.host)
        end_time = time.time()
        processing_time = end_time - start_time
        
        filename = Path(args.image).name
        
        # Enrich data with metadata
        full_result = {
            filename: {
                "image_path": args.image,
                "filename": filename,
                "street_number": data.get("street_number"),
                "street_name": data.get("street_name"),
                "unit_number": data.get("unit_number"),
                "confidence": 1.0,  # Placeholder as LLMs usually don't give this easily
                "processing_time": round(processing_time, 3)
            }
        }
        
        if args.benchmark:
            # Format: [Model] filename: JSON
            print(f"[{args.model}] {filename}: {json.dumps(full_result, indent=2)}")
        else:
            print(json.dumps(full_result[filename], indent=2))
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
