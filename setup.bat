@echo off
REM One-time setup: create venv and install dependencies (Windows)
cd /d "%~dp0"

if exist venv\Scripts\python.exe (
    echo Virtual environment already exists.
) else (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo Failed to create venv. Make sure Python is installed and on PATH.
        pause
        exit /b 1
    )
)

echo Installing packages (this may take several minutes; PyTorch is large)...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install --timeout 600 -r requirements.txt
if errorlevel 1 (
    echo Install failed. Try running setup.bat again.
    pause
    exit /b 1
)

echo.
echo Setup complete. You can now run:
echo   run_all.bat       - fetch data, train, and get a recommendation
echo   fetch.bat         - fetch book data only
echo   train.bat         - train the model only
echo   recommend.bat     - get a book recommendation
echo.
pause
