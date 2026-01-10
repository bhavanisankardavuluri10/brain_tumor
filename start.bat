@echo off
echo ================================================
echo  Starting Brain Tumor Detection System
echo ================================================
echo.

echo Starting Backend API Server...
start "Backend API" cmd /k "venv\Scripts\activate.bat && cd backend && python app.py"

timeout /t 5 /nobreak > nul

echo Starting Frontend Application...
start "Frontend" cmd /k "cd frontend && npm start"

echo.
echo ================================================
echo  System Started!
echo ================================================
echo.
echo Backend API: http://localhost:5000
echo Frontend UI: http://localhost:3000
echo.
echo Press any key to close this window (servers will keep running)...
pause > nul
