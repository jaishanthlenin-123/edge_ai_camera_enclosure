# Training Pipeline & Dataset Evolution Report

**Project:** Raspberry Pi Camera Enclosure Detection using Edge AI  
**Dataset Host:** Roboflow (`jaishanth-s-workspace/m1_m2`)  
**Trained Model:** Version 3 (`jaishanth-s-workspace/m1_m2-3-yolo26n-t1`)  

---

## 1. Dataset Evolution (M1 to M2)

| Phase | Total Source Images | Positive Enclosures | Negative Backgrounds | Augmented Size | Major Challenge Addressed |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Initial Roboflow v1** | 124 | 124 (100%) | 0 (0%) | 124 | Severe label errors (finger/background labeled instead of enclosure) |
| **Roboflow v2** | 124 | 124 (100%) | 0 (0%) | 124 | Corrected bounding boxes; wall/curtains triggered false positive (73.2% conf) |
| **User Update (Phone Uploads)** | 139 | 139 (100%) | 0 (0%) | 139 | Added 15 diverse perspective images of the enclosure |
| **Roboflow v3 (Final)** | **159** | **139** | **20 (12.6%)** | **274** | **Permanently eliminated 73.2% wall/curtain bug by injecting true negative background images** |

---

## 2. Dataset Split Breakdown (Version 3)

```
========================================================================
                      ROBOFLOW VERSION 3 SPLIT SUMMARY
========================================================================
Split      Source Images     Augmented Images    Percentage   Role
------------------------------------------------------------------------
Train      115 (100 pos + 15 neg)   230            83.9%     Model Weights Optimization
Valid       30 (27 pos + 3 neg)      30            10.9%     Validation Checkpoint Selection
Test        14 (12 pos + 2 neg)      14             5.1%     Independent Evaluation Benchmark
------------------------------------------------------------------------
Total      159 (139 pos + 20 neg)   274           100.0%     Complete Dataset
```

---

## 3. Preprocessing & Augmentation Strategy

### Aspect-Ratio-Preserving Preprocessing
- **Auto-Orient:** Enabled (`True`) to ensure correct orientation across mobile phone cameras.
- **Resize:** `640 × 640` with **Fit (black edges) in** (letterboxing).
  - *Engineering rationale:* Stretching or cropping alters the square aspect ratio of the camera enclosure and degrades lens roundness. Letterboxing preserves the true physical dimensions.

### Conservative Augmentation
- **Horizontal Flip:** Enabled ($p = 0.5$).
- **Rotation:** Small rotation ($\pm 10^\circ$).
- **Brightness & Exposure:** $\pm 15\%$ (replicates indoor day/night lighting).
- **Saturation:** $\pm 15\%$.
- **Aggressive Mosaics / Heavy Blur:** Disabled to avoid tearing the small lens features.

---

## 4. The 73.2% Wall/Curtain False-Positive Investigation

### Root Cause Analysis
During testing of earlier models, the detector incorrectly identified plain walls, curtains, and textured backgrounds as `RASPBERY PI CAM ENCLOSURE` with confidence up to **73.2%**.
1. **The Absence of Zero-Box Ground Truth:** In versions 1 and 2, 100% of images contained an enclosure.
2. **Loss Function Imbalance:** Without true negative images, the classification loss function never penalized anchor candidates activated by vertical fabric folds or corner shadow lines.
3. **The Solution:** 20 verified negative background images (plain walls, folded drapery, room corners, empty desks, and hands without the enclosure) were captured, uploaded without annotations (0 bounding boxes), and integrated into train, valid, and test splits.
4. **Verification:** Testing Model Version 3 against all negative scenarios confirmed a **0.0% false-positive rate** (100% background rejection).

---

## 5. Official Model Performance Metrics

```
Roboflow Version 3 Training Results (Epochs: 311):
- Validation mAP@50: 99.28%
- Validation Precision: 100.00%
- Validation Recall: 96.30%
- Test mAP@50: 99.31%
- Test Precision: 100.00%
- Test Recall: 91.70%
- F1 Score: 0.981 (at confidence threshold 0.38)
```
