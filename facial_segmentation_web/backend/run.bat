@echo off
ECHO Setting up Flask backend for Facial Segmentation app...

REM Create virtual environment if it doesn't exist
IF NOT EXIST venv (
    ECHO Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
ECHO Activating virtual environment...
CALL venv\Scripts\activate.bat

REM Install requirements
ECHO Installing requirements...
pip install -r requirements.txt

REM Run the Flask app
ECHO Starting Flask server...
python app.py 