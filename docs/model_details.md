# YOLO26 Nano Model Architecture & Specifications

**Project:** Raspberry Pi Camera Enclosure Detection using Edge AI  
**Model Name:** YOLO26 Nano (`yolo26n`)  
**Trained Checkpoint:** `models/best.pt` / `models/best.onnx`  
**Target Class:** `RASPBERY PI CAM ENCLOSURE` (ID `0`)  

---

## 1. Architectural Overview

YOLO26 Nano is an ultra-lightweight, high-efficiency convolutional neural network designed specifically for edge devices, embedded compute platforms (such as the Raspberry Pi 5), and real-time vision applications.

```
                      +-----------------------------+
                      |   Input Image (640x640x3)   |
                      +--------------+--------------+
                                     |
                      +--------------v--------------+
                      |     YOLO26 Nano Backbone    |
                      |  (CSPDarknet + Conv Modules)|
                      +--------------+--------------+
                                     |
                      +--------------v--------------+
                      |     PANet Feature Pyramid   |
                      |  (Multi-Scale Feature Fusion|
                      +--------------+--------------+
                                     |
                      +--------------v--------------+
                      |       Decoupled Head        |
                      |  (Anchor-Free Box + Cls)    |
                      +--------------+--------------+
                                     |
                      +--------------v--------------+
                      | Output Tensors (1, 5, 8400) |
                      | [x_c, y_c, w, h, class_conf]|
                      +-----------------------------+
```

---

## 2. Technical Specifications

| Parameter | Specification | Notes |
| :--- | :--- | :--- |
| **Model Architecture** | YOLO26 Nano (`yolo26n`) | Anchor-free decoupled detection head |
| **Total Layers** | 120 layers | Fused detection graph |
| **Parameter Count** | **2,375,031** (~2.38 Million) | Highly optimized memory footprint |
| **Computational Complexity** | **5.3 GFLOPs** | Capable of high-FPS edge execution |
| **Native Input Resolution** | `640 × 640 × 3` | Letterboxed (aspect ratio preserved) |
| **Output Shape** | `(1, 5, 8400)` | 8,400 multi-scale candidate anchor locations |
| **Number of Classes** | 1 | Single-class detector |
| **Class Identifier** | `0` | Label: `RASPBERY PI CAM ENCLOSURE` |
| **Weights File (PyTorch)** | `models/best.pt` | **5.17 MB** (5,419,973 bytes) |
| **Weights File (ONNX)** | `models/best.onnx` | **9.34 MB** (9,793,519 bytes) |

---

## 3. Benchmarked Metrics (Roboflow Cloud Model Version 3)

The model was evaluated against independent test sets and real-world domestic clutter:

| Metric | Validation Set (30 imgs) | Test Set (14 imgs) | Status |
| :--- | :---: | :---: | :---: |
| **mAP@50** | **99.28%** | **99.31%** | Benchmark Verified |
| **mAP@50-95** | **93.79%** | **93.13%** | Benchmark Verified |
| **Precision** | **100.00%** | **100.00%** | 0 False Positives |
| **Recall** | **96.30%** | **91.70%** | High Sensitivity |
| **F1 Score** | **0.981** | **0.956** | Optimal at `conf=0.38` |
| **Background Rejection** | **100.0%** | **100.0%** | 0 False Activations on Negatives |

---

## 4. Inference Latency Across Compute Hardware

| Hardware Platform | Inference Engine | Precision | Measured / Projected Latency | Estimated FPS |
| :--- | :--- | :---: | :---: | :---: |
| **NVIDIA RTX 3050 Laptop** | PyTorch / CUDA 12.4 | FP32 | **9.2 ms** | **~108 FPS** |
| **Intel Core Ultra 9 185H** | PyTorch CPU | FP32 | **18.4 ms** | **~54 FPS** |
| **Raspberry Pi 5 (Quad A76 @ 2.4GHz)** | ONNX Runtime (ARM64) | FP32 | **~48–65 ms** | **~15–21 FPS** |
| **Raspberry Pi 5 (Quad A76 @ 2.4GHz)** | PyTorch CPU (ARM64) | FP32 | **~75–95 ms** | **~10–13 FPS** |
| **Roboflow Serverless Cloud** | Hosted REST API | Serverless | **~950–1150 ms** (network roundtrip) | ~1 FPS |
