@echo off
REM AGUI RAG System Startup Script for Windows
REM This script starts both the backend AGUI server and frontend Next.js application

echo ============================================
echo Starting AGUI-Enabled RAG System
echo ============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if Node.js is installed
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Node.js is not installed or not in PATH
    pause
    exit /b 1
)

echo Prerequisites found
echo.

REM Check if ports are in use
netstat -an | findstr :8000 | findstr LISTENING >nul 2>&1
if %errorlevel% equ 0 (
    echo Warning: Port 8000 is already in use
    echo Please stop the process using port 8000 or change the backend port
    pause
    exit /b 1
)

netstat -an | findstr :3000 | findstr LISTENING >nul 2>&1
if %errorlevel% equ 0 (
    echo Warning: Port 3000 is already in use
    echo Please stop the process using port 3000 or change the frontend port
    pause
    exit /b 1
)

echo Ports 8000 and 3000 are available
echo.

REM Start backend server in new window
echo Starting AGUI Backend Server...
start "AGUI Backend Server" cmd /k "cd agent && python agent.py"

REM Wait for backend to initialize
echo Waiting for backend to initialize...
timeout /t 5 /nobreak >nul

REM Start frontend in new window
echo Starting Frontend Application...
start "CopilotKit Frontend" cmd /k "npm run dev"

REM Wait for frontend to initialize
echo Waiting for frontend to initialize...
timeout /t 5 /nobreak >nul

echo.
echo ============================================
echo AGUI RAG System Started Successfully!
echo ============================================
echo.
echo Backend (AGUI Server): http://localhost:8000
echo Frontend (CopilotKit): http://localhost:3000
echo.
echo Close this window and the other command windows to stop all services
echo.
pause