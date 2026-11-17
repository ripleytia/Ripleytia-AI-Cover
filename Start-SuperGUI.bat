@echo off
cd /d "%~dp0"
set "PATH=%USERPROFILE%\.gemini\antigravity\bin;%USERPROFILE%\.local\bin;%PATH%"
echo Starting Super RVC GUI...
"%~dp0venv\Scripts\python.exe" "%~dp0src\super_gui.py"
pause
