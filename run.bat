@echo off
REM Запуск AtomSpectra Waterfall Viewer.
REM Использование:  run.bat  [путь\к\файлу.n42]
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    REM Предпочитаем venv, если он создан.
    ".venv\Scripts\python.exe" -m awf %*
) else (
    REM venv нет — используем глобальный Python (модули установлены глобально).
    py -3.14 -m awf %* 2>nul || python -m awf %*
)
endlocal
