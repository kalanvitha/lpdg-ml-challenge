@echo off
setlocal

cd /d "%~dp0"

echo ========================================
echo LPDG Gateway Visit Ranking
echo ========================================
echo.

if exist "data\telemetry" (
    set "DATA_DIR=data"
) else (
    echo Challenge data was not found in the project data folder.
    echo.
    echo Please enter the full path to your challenge data folder.
    echo Example:
    echo C:\Users\YourName\Downloads\03-challenge-data\data
    echo.
    set /p DATA_DIR=Data folder path: 
)

if not exist "%DATA_DIR%\telemetry" (
    echo.
    echo ERROR: Telemetry folder not found:
    echo %DATA_DIR%\telemetry
    pause
    exit /b 1
)

echo.
echo Using data:
echo %DATA_DIR%
echo.

python baseline_3sigma_v4.py --data "%DATA_DIR%"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Prediction generation failed.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ========================================
echo Validating predictions.csv
echo ========================================
echo.

python validate_submission.py predictions.csv

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Validation failed.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ========================================
echo SUCCESS
echo predictions.csv is ready.
echo ========================================
pause