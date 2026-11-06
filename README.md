# Raspberry Pi Camera Enclosure Detection using Edge AI

An end-to-end Edge AI object detection pipeline designed to detect a **Raspberry Pi Camera Enclosure** in real time on a **Raspberry Pi 5** and desktop platforms using a fine-tuned **YOLO26 Nano** model.

[![YOLO26 Nano](https://img.shields.io/badge/Architecture-YOLO26%20Nano-blue.svg)](https://github.com/ultralytics/ultralytics)
[![Roboflow Version 3](https://img.shields.io/badge/Roboflow-Version%203%20(Trained)-brightgreen.svg)](https://app.roboflow.com/jaishanth-s-workspace/m1_m2/3)
[![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%205%20%7C%20Linux%20%7C%20Windows-orange.svg)]()
[![Validation mAP50](https://img.shields.io/badge/Validation%20mAP%4050-99.28%25-success.svg)]()
[![Test mAP50](https://img.shields.io/badge/Test%20mAP%4050-99.31%25-success.svg)]()

---

## 1. Project Overview

This project implements robust, low-latency object detection for the official Raspberry Pi camera enclosure across challenging edge conditions:
- Enclosure held in hand vs. placed on cluttered desks
- Connected flat ribbon cables and diverse viewing angles
- Variable indoor lighting (bright day vs. dim night)
- **Zero-false-positive rejection** on plain walls, folded curtains, windows, and empty hands.

---

## 2. Model & Dataset Specifications

| Attribute | Specification | Notes |
| :--- | :--- | :--- |
| **Project Name** | Raspberry Pi Camera Enclosure Detection using Edge AI | Embedded Vision / Edge AI |
| **Framework** | **YOLO26 Nano** (`yolo26n`) | Anchor-free decoupled head |
| **Source Platform** | **Roboflow Cloud** | Project: `jaishanth-s-workspace/m1_m2` |
| **Model Version** | **Version 3** | ID: `jaishanth-s-workspace/m1_m2-3-yolo26n-t1` |
| **Target Class** | `RASPBERY PI CAM ENCLOSURE` | Single-class detector (ID `0`) |
| **Initial Dataset** | 124 images | Initial audit phase |
| **Final Dataset (v3)** | **159 source images** (139 positive + 20 negative) | 274 augmented images |
| **Dataset Splits** | **Train:** 71% (115 imgs) \| **Valid:** 19% (30 imgs) \| **Test:** 10% (14 imgs) | Verified balanced splits |
| **Input Resolution** | **640 × 640** | Aspect ratio preserved (Letterbox: Fit in) |
| **Training Epochs** | **311 epochs** (Auto early-stop converged) | Cloud GPU cluster |
| **Weights File (PyTorch)** | `models/best.pt` (**5.17 MB** / 5,419,973 bytes) | SHA256: `43C1ABDDC728...` |
| **Weights File (ONNX)** | `models/best.onnx` (**9.34 MB** / 9,793,519 bytes) | SHA256: `B0C442759522...` |

---

## 3. Actual Benchmark Metrics

### Roboflow Cloud Version 3 Metrics (Official Test Results)
```
========================================================================
                      ROBOFLOW VERSION 3 METRICS SUMMARY
========================================================================
Split       Targets   Images   Precision    Recall    mAP@50    mAP@50-95
------------------------------------------------------------------------
Validation    27        30      100.00%     96.30%    99.28%     93.79%
Test          12        14      100.00%     91.70%    99.31%     93.13%
------------------------------------------------------------------------
```
- **Validation F1 Score:** `0.981` (at optimal threshold `conf=0.38`)
- **False-Positive Detections on Negative Images:** **0.0%** (100% background rejection)
- **Elimination of Wall/Curtain False Positive:** The previous model's 73.2% confidence activation on blank curtains was completely resolved by integrating 20 true negative background samples into Version 3.

---

## 4. Repository Structure

```
edge-ai-camera-enclosure/
├── README.md                      # Project overview and instructions
├── requirements.txt               # Dependencies for RPi 5 and Desktop
├── .gitignore                     # Secure git ignore excluding secrets & caches
├── config/
│   └── data.yaml                  # Model and class definitions
├── models/
│   ├── best.pt                    # Trained PyTorch YOLO26 Nano weights (5.17 MB)
│   ├── best.onnx                  # Exported ONNX graph for RPi 5 CPU (9.34 MB)
│   └── README.md                  # Model weights documentation and checksums
├── src/
│   ├── camera.py                  # Dual Picamera2 (RPi 5) and OpenCV camera capture
│   ├── detect.py                  # Real-time object detection inference script
│   └── utils.py                   # Visualizer, bounding box renderer, and HUD overlays
├── docs/
│   ├── model_details.md           # YOLO26 Nano architecture and specifications
│   ├── training_details.md        # Roboflow Version 3 training pipeline & metrics
│   └── raspberry_pi_setup.md      # Raspberry Pi 5 step-by-step setup guide
└── evidence/
    ├── M1/                        # Milestone 1 audit, error logs & corrections
    └── M2/                        # Milestone 2 benchmark results & 15-scenario tests
```

---

## 5. Quickstart: Desktop / Laptop Testing

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jaishanthlenin-123/M1_M2.git
   cd M1_M2
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run live webcam detection:**
   ```bash
   python src/detect.py --source 0
   ```

---

## 6. Quickstart: Raspberry Pi 5 Deployment

Full step-by-step instructions are available in [docs/raspberry_pi_setup.md](docs/raspberry_pi_setup.md).

```bash
# 1. Update system packages
sudo apt update && sudo apt install -y python3-pip python3-venv libgl1 libglib2.0-0

# 2. Clone repo and create virtual environment with system site packages
git clone https://github.com/jaishanthlenin-123/M1_M2.git
cd M1_M2
python3 -m venv --system-site-packages venv
source venv/bin/activate

# 3. Install requirements
pip install -r requirements.txt

# 4. Run real-time detection on Raspberry Pi Camera Module
python3 src/detect.py --source picam

# 5. Run with accelerated ONNX Runtime on Raspberry Pi 5 CPU
python3 src/detect.py --model models/best.onnx --source picam
```

---

## 7. Controls During Live Detection

- **`Q`** or **`ESC`**: Quit the application.
- **`S`**: Save the current detection frame to `evidence/M2/detections/`.
- **`+`**: Increase confidence threshold (filters false alarms).
- **`-`**: Decrease confidence threshold (enhances sensitivity).

---

## 8. License & Acknowledgements

- **Model Framework:** [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- **Dataset Hosting & Cloud Training:** [Roboflow](https://roboflow.com)
