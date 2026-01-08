# GPU Enablement: Code vs. Configuration

To use an NVIDIA GPU in the sidecar, you need **both** a small code change and a specific container configuration.

## 1. Code Changes (Python)

In the Python code, you must explicitly tell the PaddleOCR library to use the GPU.

**Modification to `pod_ocr.py`:**

```python
def get_ocr():
    global _ocr_instance
    if _ocr_instance is None:
        from paddleocr import PaddleOCR
        # Set use_gpu=True explicitly
        _ocr_instance = PaddleOCR(use_gpu=True, use_textline_orientation=True, lang='en')
    return _ocr_instance
```

- **Note**: If you set `use_gpu=True` on a machine without a GPU or without the correct CUDA libraries, the script will crash on startup.

## 2. Container Configuration (Config)

Even with the code set to `use_gpu=True`, the container cannot "see" the GPU unless you expose it from the host.

### A. Docker Compose Config

You must add the `deploy` section to your sidecar service:

```yaml
services:
  ocr-sidecar:
    image: my-python-ocr:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### B. Host Requirements (Host Config)

The machine running the container must have:

1.  **NVIDIA Drivers** installed.
2.  **NVIDIA Container Toolkit** installed (this is the bridge that lets Docker talk to the GPU).

## 3. Summary Checklist

| Layer         | Responsibility | Action                                      |
| :------------ | :------------- | :------------------------------------------ |
| **Code**      | Python Logic   | `PaddleOCR(use_gpu=True)`                   |
| **Container** | Docker Compose | `driver: nvidia` reservation.               |
| **Host**      | Infrastructure | Install NVIDIA Drivers + Container Toolkit. |

## 4. How to Verify

Run this command inside your sidecar container to see if it can see the GPU:

```bash
nvidia-smi
```

If this command returns the GPU stats, your **Configuration** is correct. If the Python script still runs slowly, double-check your **Code** initialization.
