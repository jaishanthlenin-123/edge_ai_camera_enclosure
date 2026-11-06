@echo off
title Roboflow Cloud YOLO26 Nano v3 - Live Webcam
cd /d "%~dp0raspberry_pi_enclosure_detection"
python scripts\webcam_roboflow_v3.py
pause

