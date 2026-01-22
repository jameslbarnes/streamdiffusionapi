@echo off
REM Build and push StreamDiffusion API Docker image
REM Usage: build_and_push.bat <dockerhub_username> [tag]

if "%1"=="" (
    echo Usage: build_and_push.bat ^<dockerhub_username^> [tag]
    echo Example: build_and_push.bat myusername streamdiffusion-api:v1
    exit /b 1
)

set USERNAME=%1
set TAG=%2
if "%TAG%"=="" set TAG=streamdiffusion-api:latest

set IMAGE=%USERNAME%/%TAG%

echo.
echo ============================================================
echo Building Docker image: %IMAGE%
echo ============================================================
echo.
echo This will take 15-30 minutes...
echo.

REM Build the image
docker build -t %IMAGE% .

if errorlevel 1 (
    echo.
    echo Build failed!
    exit /b 1
)

echo.
echo ============================================================
echo Pushing to Docker Hub: %IMAGE%
echo ============================================================
echo.

REM Push to Docker Hub
docker push %IMAGE%

if errorlevel 1 (
    echo.
    echo Push failed! Make sure you ran: docker login
    exit /b 1
)

echo.
echo ============================================================
echo SUCCESS!
echo ============================================================
echo.
echo Image pushed: %IMAGE%
echo.
echo To deploy on RunPod:
echo   python scripts/runpod_cmd.py deploy "NVIDIA H100 PCIe" 1024x576 %IMAGE%
echo.
echo Or set in .env:
echo   STREAMDIFFUSION_IMAGE=%IMAGE%
echo.
