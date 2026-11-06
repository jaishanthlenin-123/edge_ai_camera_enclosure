# Trained Models - Raspberry Pi Camera Enclosure Detection

This directory contains the trained weights for the **YOLO26 Nano** object detection model, trained directly on Roboflow Cloud (**Version 3**) on November 6, 2026.

---

## 1. Model Files Overview

| File | Format | Target Platform | Size | SHA256 Checksum |
| :--- | :---: | :--- | :---: | :--- |
| **[`best.pt`](./best.pt)** | PyTorch Checkpoint | Raspberry Pi 5 (PyTorch ARM64), Laptop / PC (CPU/GPU) | **5.17 MB** (5,419,973 bytes) | `43C1ABDDC72896043EA52003C7A238C3B924DCEB384673DB0475E275D8620530` |
| **[`best.onnx`](./best.onnx)** | ONNX Graph (Opset 18) | Raspberry Pi 5 (`onnxruntime`), Embedded Linux, NPU accelerators | **9.34 MB** (9,793,519 bytes) | `B0C4427595228D3E6B64ACB7C16270D0AE3E1140D0ADB25C4FAB17FF01642DA9` |

---

## 2. Model Specifications

- **Architecture:** YOLO26 Nano (`yolo26n`)
- **Number of Layers:** 120 layers
- **Total Parameters:** 2,375,031 (2.38M)
- **Computational Complexity:** 5.3 GFLOPs
- **Input Resolution:** `640 × 640 × 3` (BCHW format)
- **Output Shape:** `(1, 5, 8400)` (representing bounding boxes and class logits)
- **Number of Classes:** 1
- **Class Label:** `RASPBERY PI CAM ENCLOSURE` (Class ID `0`)
- **Training Epochs:** 311 epochs (Roboflow Cloud fine-tune)
- **Source Dataset:** Roboflow `jaishanth-s-workspace/m1_m2/3` (159 source images: 139 positive + 20 negative background images, 274 augmented)

---

## 3. Official Roboflow Benchmark Metrics

- **Validation mAP@50:** **99.28%**
- **Validation Precision:** **100.00%**
- **Validation Recall:** **96.30%**
- **Validation F1 Score:** **0.981** (optimal confidence threshold: 0.38)
- **Test mAP@50:** **99.31%**
- **Test Precision:** **100.00%**
- **Test Recall:** **91.70%**
- **False-Positive Rate on Negative Backgrounds:** **0.0%** (100% background rejection across plain walls, curtains, windows, and hands)

---

## 4. Usage Instructions

### PyTorch (Ultralytics)
```python
from ultralytics import YOLO

# Load model
model = YOLO('models/best.pt')

# Run inference on image
results = model('test.jpg', conf=0.35)
results[0].show()
```

### ONNX Runtime (Optimized for Raspberry Pi 5 CPU)
```python
from ultralytics import YOLO

# Ultralytics natively handles ONNX models
model = YOLO('models/best.onnx', task='detect')

# Run inference
results = model('test.jpg', conf=0.35)
```
