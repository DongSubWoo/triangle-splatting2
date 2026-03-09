@echo off
cd /d "T:\git\triangle-splatting2"

:: Derive output folder from .last_capture_dir.json
for /f "delims=" %%i in ('powershell -NoProfile -Command "(Get-Content 'T:\git\triangle-splatting2\Unreal_Export\.last_capture_dir.json' | ConvertFrom-Json).last_capture_dir"') do set SOURCE=%%i
for /f "delims=" %%i in ('powershell -NoProfile -Command "Split-Path -Leaf '%SOURCE%'"') do set FOLDER=%%i

set OUTPUT=output/%FOLDER%

:: Find latest iteration folder
for /f "delims=" %%i in ('powershell -NoProfile -Command "(Get-ChildItem 'T:\git\triangle-splatting2\%OUTPUT%\point_cloud' | Sort-Object Name | Select-Object -Last 1).FullName"') do set ITER_DIR=%%i

echo Checkpoint: %ITER_DIR%
echo Output:     %OUTPUT%/mesh.ply
echo.

python create_ply.py "%ITER_DIR%" --out "%OUTPUT%/mesh.ply"

pause
