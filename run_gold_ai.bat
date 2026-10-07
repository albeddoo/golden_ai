@echo off
echo Starting GOLD AI Terminal...
call venv\Scripts\activate.bat
python main.py
if errorlevel 1 (
    echo.
    echo ERROR: Application failed to start
    echo Check logs directory for details
    pause
)
