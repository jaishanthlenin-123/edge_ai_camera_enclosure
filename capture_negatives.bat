@echo off
title Capture Negative Background Samples
cd /d "%~dp0raspberry_pi_enclosure_detection"
python scripts\capture_negative_samples.py
pause

