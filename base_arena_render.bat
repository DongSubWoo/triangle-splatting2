@echo off
cd /d "T:\git\triangle-splatting2"

:: Derive output folder from .last_capture_dir.json
for /f "delims=" %%i in ('powershell -NoProfile -Command "(Get-Content 'T:\git\triangle-splatting2\Unreal_Export\.last_capture_dir.json' | ConvertFrom-Json).last_capture_dir"') do set SOURCE=%%i
for /f "delims=" %%i in ('powershell -NoProfile -Command "Split-Path -Leaf '%SOURCE%'"') do set FOLDER=%%i

set OUTPUT=output/%FOLDER%

echo Rendering %OUTPUT%...
echo.

python render.py -m "%OUTPUT%"

pause
