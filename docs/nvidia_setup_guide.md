# NVIDIA Infrastructure Setup Guide

This guide details how to prepare a Linux host (Ubuntu/Debian) to run GPU-accelerated containers for POD OCR.

## 1. Install NVIDIA Drivers

### Step 1: Identify your GPU

```bash
lspci | grep -i nvidia
```

### Step 2: Install Drivers

The easiest way on Ubuntu is using the `ubuntu-drivers` tool.

```bash
# Update package list
sudo apt update

# Find recommended drivers
ubuntu-drivers devices

# Install the recommended version (e.g., 535)
sudo apt install -y nvidia-driver-535
```

### Step 3: Reboot

```bash
sudo reboot
```

### Step 4: Verify

After rebooting, run:

```bash
nvidia-smi
```

_If you see a table with GPU names and memory usage, the driver is working._

---

## 2. Install NVIDIA Container Toolkit

This toolkit allows Docker to access the GPU.

### Step 1: Configure the Repository

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

### Step 2: Install the Toolkit

```bash
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

### Step 3: Configure Docker

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Step 4: Verify Container Access

Run a test container to check if it can see the GPU:

```bash
docker run --rm --runtime=nvidia --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

_If this prints the GPU table, your container infrastructure is ready._

---

## 3. Common Issues

- **"NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver"**: This usually means the driver isn't loaded. Try `sudo modprobe nvidia` or checking `dmesg`.
- **Secure Boot**: If Secure Boot is enabled, you must sign the NVIDIA drivers during installation (or disable Secure Boot in BIOS).
- **Version Mismatch**: Ensure your CUDA version in the container (PaddleOCR) is compatible with your host driver version. Drivers are generally backward compatible.

---

## 4. Multi-GPU Selection (Which GPU will be used?)

If your host has multiple GPUs (e.g., 2x NVIDIA T4), you can control which one the sidecar uses.

### Step 1: Identify GPU Indices

Run `nvidia-smi` on the host. Each GPU has an Index (0, 1, 2...).

```text
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  Tesla T4                       Off | 00000000:00:1E.0 Off |                    0 |
| N/A   34C    P8               9W /  70W |      0MiB / 15360MiB |      0%      Default |
+-----------------------------------------+----------------------+----------------------+
|   1  Tesla T4                       Off | 00000000:00:1F.0 Off |                    0 |
| N/A   36C    P8               9W /  70W |      0MiB / 15360MiB |      0%      Default |
+-----------------------------------------+----------------------+----------------------+
```

### Step 2: Target a Specific GPU

#### Choice A: Using Docker Compose (Recommended)

You can specify the `device_ids` in your `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          device_ids: ["1"] # Use GPU index 1 only
          capabilities: [gpu]
```

#### Choice B: Using Environment Variables

Alternatively, you can pass `CUDA_VISIBLE_DEVICES` to the container:

```yaml
environment:
  - CUDA_VISIBLE_DEVICES=1
```

- Setting `CUDA_VISIBLE_DEVICES=1` maps GPU index 1 of the host to GPU index 0 inside the container.
- The PaddleOCR library will then use the only GPU it can see.

### Step 3: Default Behavior

If you don't specify any index:

1.  **Docker** will typically pass **all** GPUs if you use `count: all`.
2.  **PaddleOCR** will default to using **GPU index 0**.
