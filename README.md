# POD Recognition Feature

A high-performance system for extracting street numbers, street names, and unit numbers from Proof of Delivery (POD) images using PaddleOCR. Designed to scale to 1 million images per day.

## 🚀 Key Features

- **Core OCR**: Accurate extraction of address components from natural photos.
- **High Throughput**: Architecture designed for GPU-accelerated batch inference.
- **Java 25 Integration**: Integration blueprints for modern Java backends using Spring Boot 4.0.1.
- **Hybrid Architecture**: Python-based OCR sidecars for seamless scaling.
- **Accuracy Verification**: Built-in verification processes to hit >95% accuracy targets.

## 🛠 Quick Start (Core Script)

### Prerequisites

- Python 3.12+
- (Optional) NVIDIA GPU for acceleration

### Installation

```bash
# Clone the repository
pip install -r requirements.txt
```

### Usage

Run the OCR logic on a single image:

```bash
python3 pod_ocr.py images/sample.jpg
```

Batch process a directory:

```bash
python3 pod_ocr.py ./images/ -w 4
```

---

## 📚 Documentation Index

The following documentation is available in the `docs/` folder:

### Architecture & Scaling

- **[Scaling Roadmap](docs/scaling_roadmap.md)**: Strategies for handling 1M+ images/day.
- **[Queue Architecture](docs/queue_architecture.md)**: Redis-based producer/consumer flow.
- **[Java Architecture](docs/architecture_java.md)**: Implementing the system with Java 25 and Spring Boot 4.0.1.
- **[Deployment Comparison](docs/deployment_strategy_comparison.md)**: Bare-Metal vs. Containerized deployment.

### Component Design

- **[Producer API](docs/producer_api_design.md)**: Ingestion endpoint and S3 integration.
- **[Consumer Worker](docs/consumer_worker_design.md)**: Job polling, batching, and GPU lifecycle.
- **[Sidecar Design](docs/sidecar_design.md)**: Architecture for Python and Triton sidecars.
- **[Python Sidecar Components](docs/python_sidecar_design.md)**: Internal logic of the FastAPI OCR wrapper.

### Infrastructure & Optimization

- **[GPU Configuration](docs/gpu_configuration.md)**: How to enable and target NVIDIA GPUs.
- **[NVIDIA Setup Guide](docs/nvidia_setup_guide.md)**: Installing drivers and the Container Toolkit.
- **[Accuracy Verification](docs/accuracy_verification.md)**: Benchmarking against ground truth datasets.
- **[Fine-tuning Guide](docs/finetuning_guide.md)**: Improving model performance for specific labels.

## 🔧 Environment Variables

The core script and sidecars support the following:

- `USE_GPU`: Set to `True` to enable NVIDIA GPU acceleration.
- `CUDA_VISIBLE_DEVICES`: Specify which GPU index to use (e.g., `0`).

## 🧪 Verification

Run unit tests for regex parsing:

```bash
pytest test_pod_ocr.py
```
