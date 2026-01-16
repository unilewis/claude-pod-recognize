import requests
import base64
import json
import sys
from pathlib import Path

DEFAULT_HOST = "http://192.168.86.193:11434"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def test_model(image_path, model):
    print(f"\nTesting model: {model}")
    try:
        base64_image = encode_image(image_path)
        
        prompt = """
        Analyze this image.
        1. Classify it as one of: shipping label, scene of address, map.
        2. Extract all visible text.
        
        Output format:
        Category: <category>
        Text: <text>
        """

        payload = {
            "model": model,
            "prompt": prompt,
            "images": [base64_image],
            "stream": False,
        }
        
        # Disable format=json
        print("Format NOT set to JSON")

        print("Sending simple structured prompt...")
        response = requests.post(f"{DEFAULT_HOST}/api/generate", json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        raw_response = result.get("response", "")
        
        print(f"RAW RESPONSE FROM {model}:")
        print("-" * 40)
        print(raw_response)
        print("-" * 40)
        
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    test_image = "images/pod1.jpg"
    test_model(test_image, "deepseek-ocr:latest")
    test_model(test_image, "qwen3-vl:8b")
