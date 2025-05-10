@echo off
REM — Пути (при необходимости скорректируйте)
set RAW_DIR=data\raw
set PROC_DIR=data\processed
set EXE=voxelize.exe
set PY=python

REM — Если папки processed нет, создаём
if not exist "%PROC_DIR%" (
    mkdir "%PROC_DIR%"
)

REM — Читаем модели по строкам из models_subset.txt
for /f "usebackq delims=" %%F in ("models_subset.txt") do (
    echo.
    echo === Processing model: %%F ===

    REM — 1) Запускаем C++ вокселизатор (запишет vox_bool.txt)
    "%EXE%" "%RAW_DIR%\%%F.txt"

    REM — 2) Переименовываем выход в модель-специфичный файл
    move /Y "%PROC_DIR%\vox_bool.txt" "%PROC_DIR%\%%F_vox_bool.txt" >nul

    REM — 3) Визуализация
    "%PY%" visualize.py ^
        "%RAW_DIR%\%%F.txt" ^
        "%PROC_DIR%\%%F_vox_bool.txt" ^
        --N 16 ^
        --cmap part_color_mapping.json ^
        --pts 1.0
)

echo.
echo Done.
pause
