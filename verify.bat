@echo off
echo ================================================
echo  Brain Tumor Detection System - Verification
echo ================================================
echo.

echo Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)
echo ✓ Python found
echo.

echo Checking Node.js installation...
node --version
if errorlevel 1 (
    echo ERROR: Node.js not found!
    echo Please install Node.js 14 or higher
    pause
    exit /b 1
)
echo ✓ Node.js found
echo.

echo Checking npm installation...
npm --version
if errorlevel 1 (
    echo ERROR: npm not found!
    pause
    exit /b 1
)
echo ✓ npm found
echo.

echo Checking project structure...

if exist "backend\model.py" (echo ✓ backend\model.py) else (echo ✗ backend\model.py MISSING)
if exist "backend\data_preprocessing.py" (echo ✓ backend\data_preprocessing.py) else (echo ✗ backend\data_preprocessing.py MISSING)
if exist "backend\train.py" (echo ✓ backend\train.py) else (echo ✗ backend\train.py MISSING)
if exist "backend\app.py" (echo ✓ backend\app.py) else (echo ✗ backend\app.py MISSING)
if exist "backend\predict.py" (echo ✓ backend\predict.py) else (echo ✗ backend\predict.py MISSING)
if exist "backend\config.yaml" (echo ✓ backend\config.yaml) else (echo ✗ backend\config.yaml MISSING)
if exist "requirements.txt" (echo ✓ requirements.txt) else (echo ✗ requirements.txt MISSING)
if exist "frontend\package.json" (echo ✓ frontend\package.json) else (echo ✗ frontend\package.json MISSING)
if exist "README.md" (echo ✓ README.md) else (echo ✗ README.md MISSING)

echo.
echo Checking directories...

if exist "backend" (echo ✓ backend/) else (echo ✗ backend/ MISSING)
if exist "frontend" (echo ✓ frontend/) else (echo ✗ frontend/ MISSING)
if exist "data" (echo ✓ data/) else (echo ✗ data/ MISSING)
if exist "models" (echo ✓ models/) else (echo ✗ models/ MISSING)
if exist "results" (echo ✓ results/) else (echo ✗ results/ MISSING)
if exist "uploads" (echo ✓ uploads/) else (echo ✗ uploads/ MISSING)

echo.
echo ================================================
echo  Verification Complete!
echo ================================================
echo.
echo If all checks passed, you can proceed with:
echo 1. Run setup.bat to install dependencies
echo 2. cd backend && python data_preprocessing.py
echo 4. cd backend && python data_preprocessing.py
echo 4. Run python train.py
echo 5. Run start.bat to launch the application
echo.
echo For detailed instructions, see QUICKSTART.md
echo.
pause
