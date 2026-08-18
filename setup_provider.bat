@echo off
echo ========================================
echo  GPUShare - Provider Setup
echo ========================================
echo.
echo [1/4] Creating Python virtual environment...
python -m venv .venv_provider
call .venv_provider\Scripts\activate.bat
echo.
echo [2/4] Installing provider dependencies...
pip install -r provider_agent\requirements.txt
echo.
echo [3/4] IMPORTANT:
echo  Copy the coordinator IP from the Consumer machine.
echo  The Consumer will display their IP when they start the coordinator.
echo.
set /p COORD_IP="Enter Consumer's IP address: "
set /p COORD_PORT="Enter port (default 8000): "
if "%COORD_PORT%"=="" set COORD_PORT=8000
set COORDINATOR_URL=http://%COORD_IP%:%COORD_PORT%
echo.
echo [4/4] Starting Provider Agent...
echo  Connecting to: %COORDINATOR_URL%
python provider_agent\agent.py --coordinator %COORDINATOR_URL%
pause
