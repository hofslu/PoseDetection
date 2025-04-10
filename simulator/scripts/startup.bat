@echo off
REM Step 1: Go to your project folder
cd /d C:\Users\lukas\Documents\projects\PoseDetection\simulator

REM Step 2: Activate your virtual environment
call .venv\Scripts\activate.bat

REM Step 3: Navigate to the script folder
cd /d C:\Users\lukas\Documents\projects\PoseDetection\simulator\scripts\

REM Step 4: Run your script
python esp32-cam-stream.py

