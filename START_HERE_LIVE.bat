@echo off
setlocal EnableExtensions
title Origination LIVE - Daily Betting
cd /d "%~dp0"

echo.
echo  ============================================================
echo    ORIGINATION  ·  LIVE BETTING TOOL
echo  ============================================================
echo.
echo    This is the ONLY file you need for daily live use.
echo    Opening the gameday UI...
echo.
echo    Tab 1: Daily Scan   (what to BET)
echo    Tab 2: Score Predictions  (info only)
echo.

if not exist "%~dp0.venv\Scripts\python.exe" (
  echo  [ERROR] Virtual environment not found: .venv\Scripts\python.exe
  pause
  exit /b 1
)

if not exist "%~dp0scripts\gameday_ui.py" (
  echo  [ERROR] UI missing: scripts\gameday_ui.py
  pause
  exit /b 1
)

"%~dp0.venv\Scripts\python.exe" "%~dp0scripts\gameday_ui.py"
set "EC=%ERRORLEVEL%"

if not "%EC%"=="0" (
  echo.
  echo  [ERROR] UI exited with code %EC%.
  pause
  exit /b %EC%
)

endlocal
exit /b 0
