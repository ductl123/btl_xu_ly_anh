@echo off
chcp 65001 >nul
cd /d "%~dp0"
if exist "..\.venv\Scripts\python.exe" (
    "..\.venv\Scripts\python.exe" gui.py
) else (
    python gui.py
)
if errorlevel 1 pause
