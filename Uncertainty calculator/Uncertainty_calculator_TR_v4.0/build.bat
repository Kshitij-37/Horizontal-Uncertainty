@echo off
REM Build WS Uncertainty Calculator TR_v4.0 (Multi-Reference, roughness distance-scaled) into a standalone .exe
REM Requires: pip install pyinstaller
REM Bundles the POST-THESIS (gated, 40-pair) model JSON to match the proximity-gated roughness feature.

set "SCRIPT_DIR=%~dp0"
set "MODEL_JSON=%SCRIPT_DIR%..\..\Post-thesis corrections\Results\ws_uncertainty_pairlevel_results.json"
set "SCRIPT_FILE=%SCRIPT_DIR%Uncertainty_calculator_TR_v4.0.py"

echo Checking for PyInstaller...
python -m PyInstaller --version >nul 2>&1
if errorlevel 1 (
    echo PyInstaller not found. Installing...
    pip install pyinstaller
)

echo.
echo Building WS_Uncertainty_Calculator_TR_v4.0.exe ...
echo.

python -m PyInstaller ^
    --onefile ^
    --windowed ^
    --name "WS_Uncertainty_Calculator_TR_v4.0" ^
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
echo Build successful - Multi-Reference Uncertainty Calculator (v4.0) ready to run
echo Executable: %SCRIPT_DIR%dist\WS_Uncertainty_Calculator_TR_v4.0.exe
echo.
echo You can distribute this single .exe file - the model JSON is bundled inside.
pause
