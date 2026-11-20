@echo off
title RoadSentinel Master Orchestrator
color 0A

echo ===============================================================================
echo             ROADSENTINEL - SMART TRAFFIC SAFETY ECOSYSTEM (IIT MANDI)
echo ===============================================================================
echo               FINAL DEMONSTRATION 1-CLICK SYSTEM LAUNCHER
echo ===============================================================================
echo.

:: Detect local IPv4 for direct LAN mode
for /f "tokens=4" %%a in ('route print^|find " 0.0.0.0"') do (
    set LOCAL_IP=%%a
    goto :ip_found
)
:ip_found
echo [*] Your Laptop Local LAN IP: %LOCAL_IP%
echo.

:: 1. Start FastAPI ML & Edge Camera Ingestion (Port 8000)
echo [1/5] Launching FastAPI ML Engine & Video Ingestion (Port 8000)...
start "RoadSentinel - 1. FastAPI ML Engine [8000]" cmd /k "cd /d "%~dp0fastapi-backend" && python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload"

timeout /t 3 /nobreak > nul

:: 2. Start Express Backend API (Port 3002)
echo [2/5] Launching Express Core Backend API (Port 3002)...
start "RoadSentinel - 2. Express Backend API [3002]" cmd /k "cd /d "%~dp0backend" && npm run dev"

timeout /t 2 /nobreak > nul

:: 3. Start Campus Safety Admin Dashboard (Port 3000)
echo [3/5] Launching Campus Safety Admin Dashboard (Port 3000)...
start "RoadSentinel - 3. Admin Dashboard [3000]" cmd /k "cd /d "%~dp0campus-safety-dashboard" && npm run dev -- -p 3000"

timeout /t 2 /nobreak > nul

:: 4. Start Citizen Public Safety Portal (Port 3001)
echo [4/5] Launching Citizen Public Reporting Portal (Port 3001)...
start "RoadSentinel - 4. Citizen Portal [3001]" cmd /k "cd /d "%~dp0campus-safety-website" && npm run dev -- -p 3001"

timeout /t 2 /nobreak > nul

:: 5. Start Cloudflare Tunnel (Global Public Access for RPi)
echo [5/5] Launching Cloudflare Tunnel for Edge Camera & Doppler Radar...
start "RoadSentinel - 5. Cloudflare Tunnel" cmd /k "echo Starting Cloudflare Tunnel... && cloudflared tunnel --protocol http2 --url http://127.0.0.1:8000"

echo.
echo ===============================================================================
echo                           ALL SERVICES ACTIVE!
echo ===============================================================================
echo  * Admin Dashboard   : http://localhost:3000
echo    - Live Camera Feed: http://localhost:3000/dashboard/camera
echo    - Speed Enforce   : http://localhost:3000/dashboard/enforcement
echo  * Citizen Portal    : http://localhost:3001
echo  * Express Backend   : http://localhost:3002/api
echo  * FastAPI Swagger   : http://localhost:8000/docs
echo -------------------------------------------------------------------------------
echo  RASPBERRY PI CONNECTION OPTIONS FOR DEMO:
echo   Option A (Same Wi-Fi / Hotspot - ZERO LAG):
echo     python3 rpi_camera_edge.py --backend http://%LOCAL_IP%:8000
echo.
echo   Option B (Cloudflare Tunnel - Any Network):
echo     Check window 'RoadSentinel - 5. Cloudflare Tunnel' for the https://*.trycloudflare.com URL
echo     python3 rpi_camera_edge.py --backend <CLOUDFLARE_URL>
echo ===============================================================================
echo.

:: Automatically open primary demonstration browser tabs after brief pause
timeout /t 4 /nobreak > nul
start http://localhost:3000/dashboard/camera
start http://localhost:3000/dashboard/enforcement
start http://localhost:3001

echo Done! Press any key to exit this launcher window (services will stay running).
pause > nul
