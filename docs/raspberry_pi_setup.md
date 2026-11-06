# Raspberry Pi 5 Deployment Guide: Edge AI Camera Enclosure Detection

This guide walks you through setting up and running real-time object detection on your **Raspberry Pi 5** using the trained **YOLO26 Nano** model (`models/best.pt` or `models/best.onnx`).

---

## 1. Hardware & Platform Overview

- **Device:** Raspberry Pi 5 (4GB / 8GB)
- **SoC:** Broadcom BCM2712 Quad-core ARM Cortex-A76 @ 2.4GHz
- **Operating System:** Raspberry Pi OS (64-bit Bookworm with desktop or Lite)
- **Camera Options:**
  - Raspberry Pi Camera Module 3 / Camera Module v2 / HQ Camera (connected to CSI port `CAM/DISP 0` or `1`)
  - Standard USB Webcam (plugged into USB 3.0 / 2.0 port)

> [!NOTE]
> **Inference Runtime Note (No TensorRT):**
> NVIDIA TensorRT is exclusively designed for NVIDIA CUDA GPUs and is **NOT supported** on the Raspberry Pi 5.
> The supported high-performance edge inference runtimes on Raspberry Pi 5 are:
> 1. **PyTorch ARM64** (`best.pt` via `ultralytics`)
> 2. **ONNX Runtime ARM64** (`best.onnx` via `onnxruntime`) — **Recommended** for fastest CPU inference (~15–20 FPS).

---

## 2. Step 1: System Preparation on Raspberry Pi 5

Open a terminal on your Raspberry Pi 5 and ensure your system packages are up-to-date:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv git libgl1 libglib2.0-0
```

### Camera Verification (Raspberry Pi Camera Module)
If using the official CSI Camera Module:
```bash
# Test camera preview for 3 seconds
rpicam-hello -t 3000
```
If the preview opens, your camera hardware and `libcamera` stack are functioning properly.

---

## 3. Step 2: Clone Repository & Create Virtual Environment

Clone the repository directly to your Raspberry Pi 5:

```bash
cd ~
git clone https://github.com/jaishanthlenin-123/M1_M2.git
cd M1_M2
```

### Virtual Environment Setup
> [!IMPORTANT]
> Raspberry Pi OS Bookworm enforces PEP 668 externally-managed environments. Always use a virtual environment with `--system-site-packages` so that the system-installed `picamera2` library is accessible inside the venv:

```bash
python3 -m venv --system-site-packages venv
source venv/bin/activate
```

---

## 4. Step 3: Install Dependencies

With the virtual environment active:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Verify that Ultralytics and OpenCV load properly:
```bash
python3 -c "from ultralytics import YOLO; import cv2; print('Dependencies loaded successfully!')"
```

---

## 5. Step 4: Run Real-Time Detection

### Option A: Using Raspberry Pi CSI Camera Module (Picamera2)
```bash
# Uses models/best.pt by default with Picamera2 backend
python3 src/detect.py --source picam
```

### Option B: Using USB Webcam
```bash
# Camera index 0
python3 src/detect.py --source 0
```

### Option C: Accelerated ONNX Runtime Mode (Highest FPS on RPi 5)
```bash
# Loads models/best.onnx via ONNX Runtime
python3 src/detect.py --model models/best.onnx --source picam
```

### Option D: Headless Mode (No Desktop / Monitor Connected)
If accessing the Raspberry Pi over SSH without an X11/Wayland display:
```bash
# Logs detections to terminal and saves snapshots when detected
python3 src/detect.py --source picam --no-view
```

---

## 6. Real-Time Controls

When the detection window is open:
- **`Q`** or **`ESC`**: Exit the application.
- **`S`**: Save the current annotated frame to `evidence/M2/detections/`.
- **`+`**: Increase confidence threshold (filters noise).
- **`-`**: Decrease confidence threshold (increases sensitivity).

---

## 7. Troubleshooting on Raspberry Pi 5

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| `Cannot open display` | Running over SSH without X11 forwarding | Use `--no-view` flag or enable SSH X11 forwarding (`ssh -X pi@raspberrypi.local`). |
| `No module named 'picamera2'` | Virtual environment isolated from system packages | Recreate venv with `--system-site-packages` flag. |
| Camera frame rate drops | Thermal throttling under heavy CPU load | Ensure an active cooling fan or heatsink is installed on the BCM2712 SoC. |
| Model inference latency > 100ms | Running unoptimized PyTorch FP32 on CPU | Switch to `--model models/best.onnx` to leverage ONNX Runtime optimizations. |
