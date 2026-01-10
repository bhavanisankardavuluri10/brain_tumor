@echo off
echo ================================================
echo  Brain Tumor Detection System - Setup Script
echo ================================================
echo.

echo [1/4] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo Error: Failed to create virtual environment
    pause
    exit /b 1
)

echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/4] Installing Python dependencies...
pip install --upgrade pip
cd backend
pip install -r requirements.txt
if errorlevel 1 (
    echo Error: Failed to install Python dependencies
    cd ..
    pause
    exit /b 1
)
cd ..

echo [4/4] Installing frontend dependencies...
cd frontend
call npm install
if errorlevel 1 (
    echo Error: Failed to install frontend dependencies
    cd ..
    pause
    exit /b 1
)
cd ..

echo.
echo ================================================
echo  Setup Complete!
echo ================================================
echo.
echo Next Steps:
echo 1. Place your dataset in data/raw/ directory
echo 2. Run: python data_preprocessing.py
echo 3. Run: python train.py
echo 4. Run: python app.py (in one terminal)
echo 5. Run: cd frontend ^&^& npm start (in another terminal)
echo.
pause
