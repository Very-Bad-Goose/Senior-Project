@echo off
REM Activate the environment (adjust path to client's)
call "C:\Users\angel\anaconda3\Scripts\activate.bat" "C:\Users\angel\anaconda3\envs\python.exe"
REM Change directory to where main.py is located
cd /d "C:\Users\angel\OneDrive\Desktop\Fall 2024\Senior-Project-main\src"

REM Run the Python script
python main.py

REM Pause to keep the console window open after running
pause
