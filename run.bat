@echo off
REM Run AtomSpectra Waterfall Viewer.
REM Usage:  run.bat  [path\to\file.n42]
REM Comments are ASCII-only: cmd.exe reads .bat in OEM codepage and corrupts UTF-8 Cyrillic.
setlocal
cd /d "%~dp0"
set "AWF_LOG=%~dp0awf_stderr.log"
if exist ".venv\Scripts\python.exe" (
    REM Prefer venv if it exists.
    set "AWF_PY=.venv\Scripts\python.exe"
) else (
    REM No venv: use global Python 3.14 (modules installed globally).
    set "AWF_PY=py -3.14"
)
REM -X faulthandler dumps native crashes (e.g. OpenGL segfaults) to stderr too.
%AWF_PY% -X faulthandler -m awf %* 2> "%AWF_LOG%"
if errorlevel 1 (
    echo.
    echo === AtomSpectra Viewer exited with an ERROR ===
    echo Traceback is shown below and saved to: "%AWF_LOG%"
    echo.
    type "%AWF_LOG%"
    echo.
    pause
)
endlocal
