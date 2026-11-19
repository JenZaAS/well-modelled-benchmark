@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Well map viewer
echo Close this window to stop the viewer.
echo.

set "PY="
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not defined PY (
  py -3 -c "import sys" >nul 2>&1
  if not errorlevel 1 set "PY=py -3"
)
if not defined PY (
  python -c "import sys" >nul 2>&1
  if not errorlevel 1 set "PY=python"
)
if not defined PY (
  echo Python 3 was not found. Install Python 3 and try again.
  pause
  exit /b 1
)

set "PYTHONPATH=%CD%\src;%CD%;%PYTHONPATH%"
%PY% -u -m wmb.viewer --launch
set "ERR=%ERRORLEVEL%"
if not "%ERR%"=="0" (
  echo.
  echo The viewer exited with an error.
  pause
)
exit /b %ERR%

