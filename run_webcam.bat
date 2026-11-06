@echo off
cd /d "%~dp0raspberry_pi_enclosure_detection"
echo ==============================================================
echo   Raspberry Pi Camera Enclosure - YOLO26 Nano Webcam Detector
echo ==============================================================
echo Launching live detection on NVIDIA RTX 3050 GPU...
python scripts\webcam_test.py %*
if errorlevel 1 pause

