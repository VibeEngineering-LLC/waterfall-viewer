@echo off
REM Запуск AtomSpectra Waterfall Viewer.
REM Использование:  run.bat  [путь\к\файлу.n42]
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo [ОШИБКА] venv не найден. Сначала выполните setup: см. README.md
    exit /b 1
)
".venv\Scripts\python.exe" -m awf %*
endlocal
