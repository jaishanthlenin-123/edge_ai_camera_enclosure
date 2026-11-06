@echo off
cd /d "%~dp0raspberry_pi_enclosure_detection"
echo Opening live camera feed...
python scripts\open_camera_alone.py %*
if errorlevel 1 pause

