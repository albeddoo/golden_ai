@echo off
echo ========================================
echo GOLD AI - Build Executable (.exe)
echo ========================================
echo.

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Checking PyInstaller installation...
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo Installing PyInstaller...
    pip install pyinstaller
)

echo.
echo Building executable...
echo This may take several minutes...
echo.

pyinstaller --onefile --windowed --name "GOLD_AI" ^
    --add-data "config;config" ^
    --add-data "assets;assets" ^
    --hidden-import=numpy ^
    --hidden-import=pandas ^
    --hidden-import=sklearn ^
    --hidden-import=xgboost ^
    --hidden-import=lightgbm ^
    main.py

if errorlevel 1 (
    echo.
    echo ERROR: Build failed
    echo Check build directory for logs
    pause
    exit /b 1
)

echo.
echo ========================================
echo Build Complete!
echo ========================================
echo.
echo Your executable is located at:
echo   dist\GOLD_AI.exe
echo.
echo You can now copy this file to any Windows computer
echo (Python installation not required on target machine)
echo.
pause
