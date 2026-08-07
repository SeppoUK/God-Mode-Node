# Installation Guide for GodModeInfinity

This guide will walk you through installing and setting up **GodModeInfinity** in **ComfyUI**.

---

## 📌 Prerequisites

Before installing **GodModeInfinity**, ensure you have the following:

1. **ComfyUI Installed**:
   - Download and set up ComfyUI from its [official repository](https://github.com/comfyanonymous/ComfyUI).
   - Follow the [ComfyUI installation guide](https://github.com/comfyanonymous/ComfyUI#installation).

2. **Python 3.9 or Higher**:
   - ComfyUI requires Python 3.9+. Verify your Python version:
     ```bash
     python --version
     ```

3. **NVIDIA GPU (Recommended)**:
   - A CUDA-enabled GPU is highly recommended for optimal performance.
   - Ensure you have the latest [NVIDIA drivers](https://www.nvidia.com/Download/index.aspx) and [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) installed.

4. **Sufficient VRAM**:
   - **Minimum**: 8GB VRAM (for 512x512 images).
   - **Recommended**: 12GB+ VRAM (for 1024x1024 images with hires fix and face detailing).

---

## 🛠️ Installation Steps

### Step 1: Clone the Repository

1. Navigate to your `ComfyUI/custom_nodes` directory:
   ```bash
   cd ComfyUI/custom_nodes