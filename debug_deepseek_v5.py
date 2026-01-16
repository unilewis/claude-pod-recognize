
import base64
import requests
import sys
import json

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def test(model):
    print(f"Testing {model} with prompt 'Extract the text in the image.'")
    b64 = encode_image("images/pod1.jpg")
    
    prompt_text = "Extract the text in the image."
    
    payload = {
        "model": model,
        "prompt": prompt_text,
        "images": [b64],
        "stream": False
    }

    try:
        print("Sending request...", end="", flush=True)
        resp = requests.post("http://192.168.86.193:11434/api/generate", json=payload, timeout=60)
        print("Done!")
        print(resp.json().get("response"))
    except Exception as e:
        print(f"\nFailed: {e}")

if __name__ == "__main__":
    test("deepseek-ocr:latest")
