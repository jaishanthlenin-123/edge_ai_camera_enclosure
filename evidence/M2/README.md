# Milestone 2 (M2) Evidence: Roboflow Cloud Retraining & Edge Validation

This directory contains evaluation benchmarks, verification records, and model comparison data for Milestone 2.

## Summary of Milestones (Model Version 3)

1. **Dataset Expansion & Negative Sample Integration:**
   - 15 additional positive images were uploaded via mobile capture, bringing positive images to 139.
   - Comprehensive audit revealed **zero negative (background-only) images** in versions 1 and 2, which was the direct cause of the **73.2% false-positive detection on walls and curtains**.
   - 20 unannotated negative images (plain walls, folded drapery, room corners, empty desks, and hands) were captured and uploaded, resulting in **159 total source images** (274 augmented).

2. **Roboflow Cloud Training Execution:**
   - Model architecture: `yolo26n` (YOLO26 Nano).
   - Training duration: 311 epochs (13.9 minutes cloud runtime).
   - Validation mAP@50: **99.28%**, Precision: **100.00%**, Recall: **96.30%**.
   - Test mAP@50: **99.31%**, Precision: **100.00%**, Recall: **91.70%**.

3. **15-Scenario Benchmark Validation:**
   - [`15_scenario_evaluation.json`](./15_scenario_evaluation.json): Side-by-side benchmark across 9 positive scenarios and 6 negative scenarios comparing Roboflow v3 (`m1_m2/3`), Roboflow v2 (`m1_m2/2`), and the local model (`yolo26n_best.pt`).
   - Results: **100% pass rate** on all 15 scenarios; **0.0% false-positive rate** on negative background images.
