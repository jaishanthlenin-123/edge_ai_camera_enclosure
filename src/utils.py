"""
Utility functions for edge detection, visualization, and system inspection.
"""

import os
import platform
import time
import cv2
import numpy as np

def get_system_info():
    """Returns platform and hardware details for logging."""
    info = {
        "os": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "is_raspberry_pi": False,
        "rpi_model": "Unknown"
    }

    # Check for Raspberry Pi model string
    if os.path.exists("/proc/device-tree/model"):
        try:
            with open("/proc/device-tree/model", "r") as f:
                model_str = f.read().strip("\x00")
                info["is_raspberry_pi"] = "Raspberry Pi" in model_str
                info["rpi_model"] = model_str
        except Exception:
            pass
    elif "arm" in platform.machine().lower() or "aarch64" in platform.machine().lower():
        info["is_raspberry_pi"] = True

    return info

def draw_detections(frame, detections, conf_threshold=0.35):
    """
    Draws professional bounding boxes and class tags on the frame.
    Args:
        frame: OpenCV BGR image
        detections: list of dicts with 'box' [x1, y1, x2, y2], 'conf', 'class_name'
        conf_threshold: minimum confidence to display
    Returns:
        annotated_frame: frame with drawn boxes
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    for det in detections:
        conf = det.get('conf', 0.0)
        if conf < conf_threshold:
            continue

        box = det.get('box', [0, 0, 0, 0])
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        class_name = det.get('class_name', 'ENCLOSURE')
        color = (0, 255, 0) # Vibrant green

        # Draw box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

        # Draw label header
        label = f"{class_name}: {conf*100:.1f}%"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        
        # Tag background
        tag_y1 = max(0, y1 - th - baseline - 8)
        tag_y2 = y1
        tag_x2 = min(w, x1 + tw + 10)
        cv2.rectangle(annotated, (x1, tag_y1), (tag_x2, tag_y2), color, -1)
        cv2.putText(annotated, label, (x1 + 5, tag_y2 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    return annotated

def draw_hud(frame, fps, latency_ms, detected_count, conf_threshold, model_name="YOLO26 Nano"):
    """
    Draws telemetry HUD banner at the top and bottom of the frame.
    """
    h, w = frame.shape[:2]
    annotated = frame.copy()

    # Semi-transparent top and bottom bars
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, 42), (20, 20, 20), -1)
    cv2.rectangle(overlay, (0, h - 30), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

    # Top Bar: Status & Hardware Telemetry
    if detected_count > 0:
        status_text = f"STATUS: DETECTED ({detected_count} Enclosure{'s' if detected_count > 1 else ''})"
        status_color = (0, 255, 0)
    else:
        status_text = "STATUS: SCANNING (No Enclosure Detected)"
        status_color = (0, 165, 255)

    cv2.putText(annotated, status_text, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

    metrics_text = f"FPS: {fps:.1f} | Latency: {latency_ms:.1f}ms | Conf: {conf_threshold:.2f}"
    (mw, mh), _ = cv2.getTextSize(metrics_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.putText(annotated, metrics_text, (max(15, w - mw - 15), 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # Bottom Bar: Model info and controls
    footer_text = f"Model: {model_name} | Controls: [Q] Quit  [S] Save  [+] [-] Conf"
    cv2.putText(annotated, footer_text, (15, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return annotated
