# Deployment Strategy: Bare-Metal vs. Containers

Since you are running everything on a single NVIDIA-accelerated EC2 instance, you have two choices for how to deploy the software.

## 1. Bare-Metal Deployment (No Containers)

In this mode, you install Java, Python, and CUDA directly onto the EC2 Linux OS.

### Advantages

- **Maximum Performance**: Zero overhead from container networking or virtualization.
- **Direct GPU Access**: You don't need the NVIDIA Container Toolkit. Python and PaddleOCR talk directly to the drivers you installed on the host.
- **Simpler Monitoring**: You can use standard tools like `htop`, `ps`, and `nvidia-smi` to see all your processes together.

### Disadvantages

- **Dependency Conflict**: You must manage competing versions of libraries (e.g., Python 3.12, Java 25, CUDA 12.x) on a single OS.
- **Harder to Scale**: To add a second EC2, you must manually install every dependency again or maintain a complex custom AMI.

---

## 2. Containerized Deployment (Recommended)

You run the Java API and the Python OCR as separate containers using Docker Compose.

### Advantages

- **Isolation**: The Python environment (with all its' complex CUDA dependencies) is completely separated from the clean Java/Spring Boot environment.
- **Repeatability**: "It works on my machine" becomes "It works on any EC2". You can spin up a new instance in minutes by just running `docker-compose up`.
- **Version Pinning**: You can swap out a Python version or an OCR model version without touching the Java setup.

### Disadvantages

- **Minor Overhead**: There is a negligible performance hit (usually <1%) for the container layer.
- **Setup Complexity**: Requires the NVIDIA Container Toolkit to bridge the GPU into the container.

---

## 3. Which one should you use?

| Goal                  | Recommended Path                                                          |
| :-------------------- | :------------------------------------------------------------------------ |
| **Development / POC** | **Bare-Metal** (Faster to iterate if you already have the drivers setup). |
| **1M/Day Production** | **Containers** (Essential for reliable scaling and managing 24/7 uptime). |

### Conclusion

Yes, the EC2 must be a **GPU-type machine** (e.g., AWS `g4dn.xlarge`). You **can** run without containers, but as the project grows to handle 1 million images, the maintenance of a "Bare-Metal" machine becomes much higher than a "Containerized" one.
