"""
Real-time object detection inference script for Raspberry Pi 5 and Desktop.
Loads trained YOLO26 Nano weights (best.pt or best.onnx) and runs detection.
"""

import os
import sys
import time
import argparse
import cv2
from ultralytics import YOLO

# Add parent directory to path so relative imports work seamlessly
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from src.camera import CameraCapture
from src.utils import get_system_info, draw_detections, draw_hud

def parse_args():
    parser = argparse.ArgumentParser(description="Raspberry Pi Camera Enclosure Real-Time Detection")
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join(project_dir, "models", "best.pt"),
        help="Path to trained model weights (.pt or .onnx)"
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Video source: camera index (0, 1), 'picam' for RPi Camera Module, or video path"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Detection confidence threshold (default: 0.35)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image resolution (default: 640)"
    )
    parser.add_argument(
        "--view",
        action="store_true",
        default=True,
        help="Display live GUI window"
    )
    parser.add_argument(
        "--no-view",
        dest="view",
        action="store_false",
        help="Run in headless mode (no GUI window)"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=os.path.join(project_dir, "evidence", "M2", "detections"),
        help="Directory to save snapshot captures"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 65)
    print("  Raspberry Pi Camera Enclosure Detection - Edge Inference")
    print("=" * 65)

    sys_info = get_system_info()
    print(f"[System] OS: {sys_info['os']} ({sys_info['machine']})")
    if sys_info['is_raspberry_pi']:
        print(f"[System] Platform: {sys_info['rpi_model']}")

    # 1. Verify and Load Model
    if not os.path.exists(args.model):
        # Check fallback to best.onnx if best.pt is specified or vice versa
        alt_model = args.model.replace(".pt", ".onnx") if args.model.endswith(".pt") else args.model.replace(".onnx", ".pt")
        if os.path.exists(alt_model):
            print(f"[Model] Specified model '{args.model}' not found. Using alternative: '{alt_model}'")
            args.model = alt_model
        else:
            print(f"[ERROR] Model file not found at: {args.model}")
            sys.exit(1)

    print(f"[Model] Loading model from: {args.model}...")
    t_load = time.time()
    try:
        model = YOLO(args.model)
        print(f"[Model] Model loaded in {(time.time() - t_load)*1000:.1f}ms")
        print(f"[Model] Target Classes: {model.names}")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)

    # 2. Open Camera Source
    print(f"[Camera] Connecting to source: '{args.source}'...")
    camera = CameraCapture(source=args.source, width=640, height=480, fps=30)

    # 3. GUI Setup
    window_name = "Edge AI - Raspberry Pi Camera Enclosure Detection"
    if args.view:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 540)

    conf_thresh = args.conf
    fps = 0.0
    prev_time = time.time()

    print("\n[Inference] Starting live detection loop...")
    print("  Controls: [Q] Quit | [S] Save Snapshot | [+] / [-] Adjust Confidence\n")

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                print("[WARNING] Could not read frame from camera.")
                time.sleep(0.05)
                continue

            # Compute FPS
            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-5, (curr_time - prev_time)))
            prev_time = curr_time

            # Run YOLO inference
            t_infer = time.time()
            results = model(frame, conf=conf_thresh, imgsz=args.imgsz, verbose=False)
            latency_ms = (time.time() - t_infer) * 1000

            # Extract detections
            detections = []
            if results and len(results) > 0:
                boxes = results[0].boxes
                for box in boxes:
                    xyxy = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = model.names.get(cls_id, "ENCLOSURE")
                    detections.append({
                        "box": xyxy,
                        "conf": conf,
                        "class_name": cls_name
                    })

            # Render visuals
            annotated = draw_detections(frame, detections, conf_threshold=conf_thresh)
            annotated = draw_hud(
                annotated,
                fps=fps,
                latency_ms=latency_ms,
                detected_count=len(detections),
                conf_threshold=conf_thresh,
                model_name="YOLO26 Nano (Roboflow v3)"
            )

            # Display or headless logging
            if args.view:
                cv2.imshow(window_name, annotated)
                key = cv2.waitKey(1) & 0xFF

                if key in [ord('q'), ord('Q'), 27]: # Q or ESC
                    print("\n[Inference] Quit requested by user.")
                    break
                elif key in [ord('s'), ord('S')]: # Save frame
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    snap_name = f"detection_{timestamp}.jpg"
                    snap_path = os.path.join(args.save_dir, snap_name)
                    cv2.imwrite(snap_path, annotated)
                    print(f"[Snapshot] Saved detection frame to: {snap_path}")
                elif key in [ord('+'), ord('=')]:
                    conf_thresh = min(0.95, conf_thresh + 0.05)
                    print(f"[Threshold] Confidence threshold increased to: {conf_thresh:.2f}")
                elif key in [ord('-'), ord('_')]:
                    conf_thresh = max(0.10, conf_thresh - 0.05)
                    print(f"[Threshold] Confidence threshold decreased to: {conf_thresh:.2f}")
            else:
                # Headless console log
                if len(detections) > 0:
                    best_conf = max([d['conf'] for d in detections])
                    print(f"[Detection] Enclosure detected! Conf: {best_conf*100:.1f}%, Latency: {latency_ms:.1f}ms, FPS: {fps:.1f}")

    except KeyboardInterrupt:
        print("\n[Inference] Interrupted by keyboard.")
    finally:
        camera.release()
        if args.view:
            cv2.destroyAllWindows()
        print("[Inference] Shutdown complete.")

if __name__ == '__main__':
    main()

