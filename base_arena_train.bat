@echo off
cd /d "T:\git\triangle-splatting2"

:: Read source path and derive output folder name from .last_capture_dir.json
for /f "delims=" %%i in ('powershell -NoProfile -Command "(Get-Content 'T:\git\triangle-splatting2\Unreal_Export\.last_capture_dir.json' | ConvertFrom-Json).last_capture_dir"') do set SOURCE=%%i
for /f "delims=" %%i in ('powershell -NoProfile -Command "Split-Path -Leaf '%SOURCE%'"') do set FOLDER=%%i

set OUTPUT=output/%FOLDER%

echo Source: %SOURCE%
echo Model:  %OUTPUT%
echo.

python train.py -s "%SOURCE%" -m "%OUTPUT%" ^
    --eval ^
    --iterations 30000 ^
    --save_iterations 5000 10000 15000 20000 25000 30000 ^
    --test_iterations 5000 10000 15000 20000 25000 30000 ^
    --resolution 1 ^
    --sh_degree 0 ^
    --data_device cpu ^
    --use_normal

pause
