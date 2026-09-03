@echo off
REM Build WS Uncertainty Calculator TR_FinalModel into a standalone .exe.
REM Un-gated, single model: bundles the ORIGINAL thesis-final JSON byte-for-byte (no re-fit).
REM Requires: pip install pyinstaller

set "SCRIPT_DIR=%~dp0"
set "MODEL_JSON=%SCRIPT_DIR%..\..\Bayesian_approach\Final model\Results\ws_uncertainty_pairlevel_results.json"
set "SCRIPT_FILE=%SCRIPT_DIR%Uncertainty_calculator_TR_FinalModel.py"

if not exist "%MODEL_JSON%" (
    echo ERROR: %MODEL_JSON% not found.
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
echo Building WS_Uncertainty_Calculator_TR_FinalModel.exe (exact thesis-final model) ...
echo.

python -m PyInstaller ^
    --onefile ^
    --windowed ^
    --name "WS_Uncertainty_Calculator_TR_FinalModel" ^
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
echo Build successful - exact thesis-final calculator ready.
echo Executable: %SCRIPT_DIR%dist\WS_Uncertainty_Calculator_TR_FinalModel.exe
pause
