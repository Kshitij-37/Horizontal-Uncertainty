@echo off
REM Build WS Uncertainty Calculator TR_v5 into a standalone .exe.
REM Sigma-only, adaptive roughness, n=47 corrected-data fit.
REM Bundles the Adaptive Model v5 JSON from
REM   Post-thesis corrections/Adaptive_Model_v5/Results/model_results.json
REM Requires: pip install pyinstaller

set "SCRIPT_DIR=%~dp0"
REM Prefer the local copy (kept in sync with the source-of-truth)
set "MODEL_JSON=%SCRIPT_DIR%ws_uncertainty_expM2c_results.json"
if not exist "%MODEL_JSON%" (
    set "MODEL_JSON=%SCRIPT_DIR%..\..\Post-thesis corrections\Adaptive_Model_v5\Results\model_results.json"
)
set "SCRIPT_FILE=%SCRIPT_DIR%Uncertainty_calculator_TR_v5.py"

if not exist "%MODEL_JSON%" (
    echo ERROR: %MODEL_JSON% not found.
    echo Copy the model JSON into this folder, or re-run
    echo   Post-thesis corrections/Adaptive_Model_v5/model_script.py
    echo to regenerate it.
    pause
    exit /b 1
)

echo Checking for PyInstaller...
python -m PyInstaller --version >nul 2>&1
if errorlevel 1 (
    echo PyInstaller not found. Installing...
    pip install pyinstaller
)

echo.
echo Building WS_Uncertainty_Calculator_TR_v5.exe (adaptive roughness, sigma-only) ...
echo.

python -m PyInstaller ^
    --onefile ^
    --windowed ^
    --name "WS_Uncertainty_Calculator_TR_v5" ^
    --add-data "%MODEL_JSON%;." ^
    --distpath "%SCRIPT_DIR%dist" ^
    --workpath "%SCRIPT_DIR%build" ^
    --specpath "%SCRIPT_DIR%." ^
    "%SCRIPT_FILE%"

if errorlevel 1 (
    echo.
    echo BUILD FAILED.
    pause
    exit /b 1
)

echo.
echo Build successful - TR_v5 calculator ready.
echo Executable: %SCRIPT_DIR%dist\WS_Uncertainty_Calculator_TR_v5.exe
pause
