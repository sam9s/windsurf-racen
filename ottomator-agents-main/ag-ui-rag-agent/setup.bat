@echo off
REM Setup script for AGUI RAG System on Windows

echo ============================================
echo AGUI RAG System Setup
echo ============================================
echo.

echo This script will install all required dependencies.
echo Make sure you have Python 3.9+ and Node.js 18+ installed.
echo.
pause

REM Install Python dependencies
echo.
echo Installing Python dependencies...
cd agent
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Error installing Python dependencies
    pause
    exit /b 1
)
cd ..

REM Install Node.js dependencies
echo.
echo Installing Node.js dependencies...
npm install
if %errorlevel% neq 0 (
    echo Error installing Node.js dependencies
    pause
    exit /b 1
)

echo.
echo ============================================
echo Setup completed successfully!
echo ============================================
echo.
echo Next steps:
echo 1. Configure your .env file in the agent directory
echo 2. Initialize the database: python -m agent.utils.db_utils init
echo 3. Ingest documents (optional): python -m agent.ingestion.ingest --documents ./documents
echo 4. Run the system: start_agui_system.bat
echo.
pause