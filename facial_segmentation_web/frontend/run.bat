@echo off
ECHO Setting up React frontend for Facial Segmentation app...

REM Check if node_modules exists
IF NOT EXIST node_modules (
    ECHO Installing dependencies...
    CALL npm install
)

REM Start the development server
ECHO Starting the Vite development server...
CALL npm run dev 