@echo off
title News2Trade AI
echo.
echo  =============================================
echo     News2Trade AI - Starting...
echo  =============================================
echo.

:: Try python3.13 first (Microsoft Store), then python
where python3.13.exe >nul 2>&1
if %errorlevel%==0 (
    set PYTHON=python3.13.exe
) else (
    where python >nul 2>&1
    if %errorlevel%==0 (
        set PYTHON=python
    ) else (
        echo [ERROR] Python not found. Please install Python 3.10+
        pause
        exit /b 1
    )
)

echo  Using: %PYTHON%
echo.

:: Install dependencies if needed
echo  [1/2] Checking dependencies...
%PYTHON% -m pip install -r requirements.txt -q 2>nul

:: Launch Streamlit
echo  [2/2] Launching app in browser...
echo.
echo  -----------------------------------------------
echo   Open http://localhost:8501 if it doesn't open
echo  -----------------------------------------------
echo.
%PYTHON% -m streamlit run app.py --server.headless true
pause
