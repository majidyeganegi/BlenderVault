@echo off
REM ===============================
REM BlenderVault Auto Dependencies Installer
REM ===============================
echo Searching for Blender installation...

REM --- Step 1: Try to detect Blender from registry ---
for /f "tokens=2*" %%a in ('reg query "HKLM\SOFTWARE\BlenderFoundation\Blender" /v Install_Dir 2^>nul') do (
    set BLENDER_PATH=%%b
)

REM --- Step 2: If not found in registry, try default path ---
if not defined BLENDER_PATH (
    set BLENDER_PATH="E:\Program Files\Blender Foundation\Blender 4.5"
)

REM --- Step 3: Check if Blender exists ---
if exist %BLENDER_PATH% (
    echo Blender found at %BLENDER_PATH%
) else (
    echo Blender not found at defined path!
    echo Please edit this script to point to your Blender folder.
    pause
    exit /b
)

REM --- Step 4: Set Blender Python paths ---
REM Adjust '4.5' if your version folder is different inside the Blender path
set PYTHON_DIR=%BLENDER_PATH%\4.5\python
set PYTHON_BIN=%PYTHON_DIR%\bin

REM --- Step 5: Upgrade pip ---
echo Upgrading pip...
"%PYTHON_BIN%\python.exe" -m ensurepip
"%PYTHON_BIN%\python.exe" -m pip install --upgrade pip

REM --- Step 6: Install Required Science Libraries ---
REM Removing version caps to ensure compatibility with Blender 4.5's newer Python
echo Installing Scientific Libraries...

echo Installing Numpy...
"%PYTHON_BIN%\python.exe" -m pip install numpy

echo Installing Scipy...
"%PYTHON_BIN%\python.exe" -m pip install scipy

REM --- Step 7: Verification ---
echo.
echo Verifying Installation...
"%PYTHON_BIN%\python.exe" -c "import scipy; import scipy.spatial; import scipy.sparse; print('SciPy Version:', scipy.__version__, '- SUCCESS')"

echo.
echo All dependencies installed successfully!
pause