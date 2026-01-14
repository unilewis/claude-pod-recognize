
import base64
import requests
import sys
import json

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def test(model, format_json=False):
    print(f"Testing {model} with format_json={format_json}")
    b64 = encode_image("test_small.jpg")
    payload = {
        "model": model,
        "prompt": "Extract",
        "images": [b64],
        "stream": False
    }
    if format_json:
        payload["format"] = "json"

    try:
        print("Sending request...", end="", flush=True)
        resp = requests.post("http://192.168.86.193:11434/api/generate", json=payload, timeout=60)
        print("Done!")
        print(resp.json().get("response"))
    except Exception as e:
        print(f"\nFailed: {e}")

if __name__ == "__main__":
    test("ministral-3:8b", format_json=False)
