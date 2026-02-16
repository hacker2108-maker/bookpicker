@echo off
REM Run full pipeline: fetch data, train, recommend (Windows)
cd /d "%~dp0"

if not exist venv\Scripts\python.exe (
    echo Run setup.bat first to create the environment and install dependencies.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

echo [1/3] Fetching book data...
python fetch_data.py
if errorlevel 1 (
    echo Fetch failed.
    pause
    exit /b 1
)

echo.
echo [2/3] Training model...
python train.py
if errorlevel 1 (
    echo Train failed.
    pause
    exit /b 1
)

echo.
echo [3/3] Recommendation:
python recommend.py

echo.
pause
