@echo off
echo ================================================
echo  Running Prediction
echo ================================================
echo.

if "%~1"=="" (
    echo Usage: run_predict.bat path\to\image.jpg
    echo.
    echo Options:
    echo   --image path\to\image.jpg       Predict single image
    echo   --directory path\to\folder      Predict all images in folder
    echo   --visualize                     Show visualization
    echo.
    echo Example:
    echo   run_predict.bat --image test.jpg --visualize
    pause
    exit /b 1
)

cd /d "%~dp0"
if not exist "..\venv\Scripts\activate.bat" (
    echo Error: Virtual environment not found!
    echo Please run setup.bat from the root directory first.
    pause
    exit /b 1
)

call ..\venv\Scripts\activate.bat
python predict.py %*

pause
