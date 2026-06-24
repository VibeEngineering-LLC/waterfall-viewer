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
    set "AWF_PIP_USER="
) else (
    REM No venv: use global Python 3.14 with per-user site-packages.
    set "AWF_PY=py -3.14"
    set "AWF_PIP_USER=--user"
)
REM First-run self-heal: if a dependency is missing, install requirements into
REM the user's own site-packages (no admin needed). Runs in the user's real
REM environment, so packages land where this same Python will look for them.
%AWF_PY% -c "import numpy, PySide6, pyqtgraph, OpenGL, lxml" 1>nul 2>nul
if errorlevel 1 (
    echo Installing dependencies on first run, please wait...
    %AWF_PY% -m pip install %AWF_PIP_USER% -r requirements.txt
    if errorlevel 1 (
        echo.
        echo === Could not install dependencies automatically ===
        echo Run manually:  py -3.14 -m pip install --user -r requirements.txt
        echo.
        pause
        endlocal
        exit /b 1
    )
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
