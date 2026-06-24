@echo off
REM Run AtomSpectra Waterfall Viewer.
REM Usage:  run.bat  [path\to\file.n42]
REM Comments are ASCII-only: cmd.exe reads .bat in OEM codepage and corrupts UTF-8 Cyrillic.
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    REM Prefer venv if it exists.
    ".venv\Scripts\python.exe" -m awf %*
) else (
    REM No venv: use global Python 3.14 (modules installed globally).
    py -3.14 -m awf %*
)
endlocal
