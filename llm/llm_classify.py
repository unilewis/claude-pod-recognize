#!/usr/bin/env python3
"""
LLM Image Classifier via Ollama

Classifies images into one of the following categories:
- shipping label
- scene of address
- map

Uses local Multimodal LLMs via the Ollama API.
Default Model: gemma3:4b
"""

import argparse
import base64
import json
import sys
import requests
from pathlib import Path
import time

# Default configuration
DEFAULT_MODEL = "gemma3:4b"
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

def classify_image(image_path, model, host=DEFAULT_HOST):
    """Sends image to Ollama for classification."""
    
    base64_image = encode_image(image_path)
    
    # Special handling for models that struggle with strict JSON or complex instructions
    use_simple_prompt = "deepseek" in model or "qwen" in model
    
    if use_simple_prompt:
        prompt = """
        Analyze this image.
        1. Classify it as exactly one of: shipping label, scene of address, map.
        
        Output format:
        Category: <category>
        """
    else:
        prompt = f"""
        Analyze this image and classify it into exactly one of the following categories:
        - shipping label
        - scene of address
        - map
        
        Return ONLY a valid JSON object with the following keys:
        - category (string, must be one of the above)
        - confidence (float, between 0.0 and 1.0)
        
        Do not include markdown formatting.
        """

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [base64_image],
        "stream": True,
    }

    # Only force JSON format for standard models
    if not use_simple_prompt:
        payload["format"] = "json"
    
    try:
        # Use streaming to avoid timeouts on slow models, but allow long initial wait (queue)
        with requests.post(f"{host}/api/generate", json=payload, stream=True, timeout=30) as response:
            try:
                response.raise_for_status()
            except requests.exceptions.HTTPError as e:
                # Provide more context if possible
                raise RuntimeError(f"API Error: {e.response.text}") from e
                
            raw_response_parts = []
            for line in response.iter_lines():
                if line:
                    part = json.loads(line)
                    if "response" in part:
                        raw_response_parts.append(part["response"])
                    if part.get("done"):
                        break
            
            raw_response = "".join(raw_response_parts)
        
        if use_simple_prompt:
            # Parse Key-Value format
            import re
            
            # More robust regex: Look for "Category" at start of line
            category_match = re.search(r"(?:^|\n)\*?\**Category\*?\**:\s*(.+)", raw_response, re.IGNORECASE)
            
            # Look for "Text" or "Extracted"
            # text_match = re.search(r"(?:^|\n)\*?\**Text\*?\**:\s*(.*)", raw_response, re.IGNORECASE | re.DOTALL)
            
            raw_cat = category_match.group(1).strip() if category_match else "unknown"
            # Normalize category
            found_cat = "unknown"
            
            # Clean punctuation from category (e.g. "shipping label.")
            clean_cat = raw_cat.rstrip(".,*").lower()
            
            for c in CATEGORIES:
                if c in clean_cat:
                    found_cat = c
                    break
            
            data = {
                "category": found_cat,
                "confidence": 0.85, 
                "full_text": ""
            }
        else:
            # Clean up potential markdown code blocks
            cleaned_response = raw_response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
                
            data = json.loads(cleaned_response)
        
        # Validation
        if data.get("category") not in CATEGORIES:
             pass
             
        return data

    except requests.exceptions.ConnectionError:
        raise ConnectionError(f"Could not connect to Ollama at {host}. Is it running?")
    except json.JSONDecodeError:
        # Fallback for JSON models that failed
        print(f"JSON Error for {model}, falling back to raw.", file=sys.stderr)
        return {
            "category": "unknown",
            "confidence": 0.0,
            "full_text": raw_response
        }
    except Exception as e:
        raise RuntimeError(f"Classification failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Classify image using local LLMs via Ollama.")
    parser.add_argument("--image", required=True, help="Path to the image file")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Ollama API host")
    
    args = parser.parse_args()
    
    try:
        start_time = time.time()
        result = classify_image(args.image, args.model, args.host)
        end_time = time.time()
        
        # Add metadata
        result["filename"] = Path(args.image).name
        result["processing_time"] = round(end_time - start_time, 3)
        
        print(json.dumps(result, indent=2))
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
