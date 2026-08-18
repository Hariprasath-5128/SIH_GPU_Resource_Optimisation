@echo off
echo ========================================
echo  GPUShare - Consumer Setup
echo ========================================
echo.
echo [1/4] Creating Python virtual environment...
python -m venv .venv_coordinator
call .venv_coordinator\Scripts\activate.bat
echo.
echo [2/4] Installing coordinator dependencies...
pip install -r coordinator\requirements.txt
echo.
echo [3/4] Starting GPUShare Coordinator...
echo  Coordinator will run at http://localhost:8000
echo  Open VS Code and run: GPUShare: Open Consumer Dashboard
echo.
echo [4/4] Launch coordinator:
python coordinator\main.py
echo.
pause
