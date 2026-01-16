#!/usr/bin/env python3
"""
Combined LLM Image Classifier and Extractor via Ollama

Performs two tasks in a single pass:
1. Classifies image (shipping label, scene of address, map)
2. Extracts address details (street number, street name, unit number)

Uses local Multimodal LLMs via the Ollama API.
Default Model: qwen3-vl:8b
"""

import argparse
import base64
import json
import sys
import requests
from pathlib import Path
import time

# Default configuration
DEFAULT_MODEL = "qwen3-vl:8b"
DEFAULT_HOST = "http://localhost:11434"

CATEGORIES = [
    "shipping label",
    "scene of address",
    "map"
]

def encode_image(image_path):
    """Encodes an image to base64."""
    if not Path(image_path).exists():
        print(f"Error: Image not found at {image_path}", file=sys.stderr)
        sys.exit(1)
        
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def process_image(image_path, model, host=DEFAULT_HOST):
    """Sends image to Ollama for combined classification and extraction."""
    
    base64_image = encode_image(image_path)
    
    # Single prompt for both tasks
    prompt = """
    Analyze this image.
    
    Task 1: Classify it into exactly one of the following categories:
    - shipping label
    - scene of address
    - map
    
    Task 2: Extract the delivery address details if present:
    - street_number (string or null)
    - street_name (string or null)
    - unit_number (string or null)
    
    Return ONLY a valid JSON object with the following keys:
    {
        "category": "category_name",
        "street_number": "number",
        "street_name": "name",
        "unit_number": "unit"
    }
    
    Do not include markdown formatting or explanations.
    """

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [base64_image],
        "stream": False
    }
    
    try:
        response = requests.post(f"{host}/api/generate", json=payload, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        raw_response = result.get("response", "")
        
        # Clean up potential markdown code blocks
        cleaned_response = raw_response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
            
        data = json.loads(cleaned_response)
        
        # Normalize category
        raw_cat = data.get("category", "unknown").lower()
        found_cat = "unknown"
        for c in CATEGORIES:
            if c in raw_cat:
                found_cat = c
                break
        data["category"] = found_cat
        
        return data

    except requests.exceptions.ConnectionError:
        raise ConnectionError(f"Could not connect to Ollama at {host}. Is it running?")
    except json.JSONDecodeError:
        # Fallback for invalid JSON
        print(f"JSON Error for {model}, falling back to defaults.", file=sys.stderr)
        return {
            "category": "unknown",
            "street_number": None,
            "street_name": None,
            "unit_number": None,
            "raw_output": raw_response
        }
    except Exception as e:
        raise RuntimeError(f"Processing failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Classify & Extract from image using local LLMs.")
    parser.add_argument("--image", required=True, help="Path to the image file")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Ollama API host")
    
    args = parser.parse_args()
    
    try:
        start_time = time.time()
        result = process_image(args.image, args.model, args.host)
        end_time = time.time()
        
        # Add metadata
        result["filename"] = Path(args.image).name
        result["processing_time"] = round(end_time - start_time, 3)
        result["model"] = args.model
        
        print(json.dumps(result, indent=2))
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
