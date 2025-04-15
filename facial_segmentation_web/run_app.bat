@echo off
ECHO Starting Facial Segmentation Web Application...

REM Start the backend in a new window
ECHO Starting backend server...
start cmd /k "cd backend && run.bat"

REM Wait for backend to initialize
ECHO Waiting for backend to initialize...
timeout /t 5 /nobreak > nul

REM Start the frontend in a new window
ECHO Starting frontend application...
start cmd /k "cd frontend && run.bat"

ECHO Application started! Frontend will be available at http://localhost:3000
ECHO Press any key to stop all services...
pause > nul

REM Clean up by killing processes (in a real production environment, you would want a more graceful shutdown)
taskkill /f /im python.exe > nul 2>&1
taskkill /f /im node.exe > nul 2>&1

ECHO Application stopped. 