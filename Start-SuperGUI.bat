@echo off
cd /d "%~dp0"
echo Starting Super RVC GUI...
call venv\Scripts\activate.bat
python src\super_gui.py
pause
