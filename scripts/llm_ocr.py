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
    "llama4:scout"
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
    prompt = """
    Analyze this image and extract the delivery address details.
    Return ONLY a valid JSON object with the following keys:
    - street_number (string or null)
    - street_name (string or null)
    - unit_number (string or null)

    If a field is not visible, use null. do not include markdown formatting like ```json.
    """

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [base64_image],
        "stream": False,
        "format": "json"  # Force JSON mode if supported by the model
    }

    try:
        response = requests.post(f"{host}/api/generate", json=payload)
        response.raise_for_status()
        
        result = response.json()
        raw_response = result.get("response", "")
        
        # Clean up potential markdown code blocks if the model ignores the "format: json" sometimes
        cleaned_response = raw_response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
            
        return json.loads(cleaned_response)

    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to Ollama at {host}. Is it running?", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Model did not return valid JSON. Raw output:\n{raw_response}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Extract address from image using local LLMs via Ollama.")
    parser.add_argument("--image", required=True, help="Path to the image file")
    parser.add_argument("--model", default="qwen3-vl:8b", help=f"Ollama model to use. Options: {', '.join(SUPPORTED_MODELS)}")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama API host")
    
    args = parser.parse_args()
    
    print(f"Processing {args.image} with {args.model}...")
    data = extract_address(args.image, args.model, args.host)
    
    print(json.dumps(data, indent=2))

if __name__ == "__main__":
    main()
