@echo off
chcp 65001 >nul
title He Thong Nhan Dien Bien So Xe - Digital Image Processing
cd /d "%~dp0LicensePlateDetection"

echo ========================================================
echo  DANG KHOI CHAY GIAO DIEN DO HOA (DESKTOP GUI)...
echo ========================================================

if exist "..\.venv\Scripts\python.exe" (
    ..\.venv\Scripts\python.exe gui.py
) else (
    py gui.py
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [!] Gap loi khi chay. Nhan phim bat ky de thoat...
    pause >nul
)

