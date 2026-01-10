@echo off
echo ================================================
echo  Starting Backend API Server
echo ================================================
echo.

cd /d "%~dp0"
if not exist "..\venv\Scripts\activate.bat" (
    echo Error: Virtual environment not found!
    echo Please run setup.bat from the root directory first.
    pause
    exit /b 1
)

call ..\venv\Scripts\activate.bat
python app.py

pause
