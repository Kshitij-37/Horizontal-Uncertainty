@echo off
REM Build WS Uncertainty Calculator TR_v3.0 (un-gated, with model dropdown) into a standalone .exe
REM Requires: pip install pyinstaller
REM PREREQUISITE: run build_dropdown_models.py (in pymc-env) first to generate the two model JSONs.

set "SCRIPT_DIR=%~dp0"
set "MODEL_CONS=%SCRIPT_DIR%model_conservative_results.json"
set "MODEL_LOW=%SCRIPT_DIR%model_lowfloor_results.json"
set "SCRIPT_FILE=%SCRIPT_DIR%Uncertainty_calculator_TR_v3.0.py"

if not exist "%MODEL_CONS%" (
    echo ERROR: %MODEL_CONS% not found.
    echo Run build_dropdown_models.py in pymc-env first to generate the two model JSONs.
    pause
    exit /b 1
)
if not exist "%MODEL_LOW%" (
    echo ERROR: %MODEL_LOW% not found.
    echo Run build_dropdown_models.py in pymc-env first to generate the two model JSONs.
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
echo Building WS_Uncertainty_Calculator_TR_v3.0.exe (both models bundled) ...
echo.

python -m PyInstaller ^
    --onefile ^
    --windowed ^
    --name "WS_Uncertainty_Calculator_TR_v3.0" ^
    --add-data "%MODEL_CONS%;." ^
    --add-data "%MODEL_LOW%;." ^
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
echo Build successful - dropdown calculator ready (Conservative / Low-floor models bundled).
echo Executable: %SCRIPT_DIR%dist\WS_Uncertainty_Calculator_TR_v3.0.exe
pause
