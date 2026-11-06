# Milestone 1 (M1) Evidence: Dataset Audit & Quality Assurance

This directory contains the verification records and root-cause analysis for Milestone 1.

## Summary of Findings (Initial 124 Images)

1. **Initial Model Failure Root Cause:**
   - The initial Roboflow model had significant bounding box misalignment (boxes enclosing fingers, empty background, or table corners instead of the camera enclosure).
   - Class label was initially recorded in Roboflow as `'RASPBERY PI CAM ENCLOSURE'`.
2. **Audit Artifacts:**
   - [`dataset_audit_report.md`](./dataset_audit_report.md): Detailed image-by-image inspection of all 124 original images.
   - [`annotation_errors.csv`](./annotation_errors.csv): Tabular log of identified box misalignments, loose bounds, and background errors.
   - [`class_distribution.csv`](./class_distribution.csv): Split breakdown across train (88), valid (24), and test (12).
3. **Correction:**
   - Ground truth annotations were cleaned and tightened locally and verified against real-world test scenes before progressing to Milestone 2.

