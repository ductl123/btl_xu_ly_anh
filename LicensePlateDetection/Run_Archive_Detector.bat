@echo off
setlocal
cd /d "%~dp0"
if not exist "..\.venv\Scripts\python.exe" (
  echo Khong tim thay moi truong Python cua du an.
  pause
  exit /b 1
)
"..\.venv\Scripts\python.exe" tools\detect_archive_plates.py %*
if errorlevel 1 pause
