#!/bin/bash

# AGUI RAG System Startup Script
# This script starts both the backend AGUI server and frontend Next.js application

echo "============================================"
echo "Starting AGUI-Enabled RAG System"
echo "============================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if a port is in use
port_in_use() {
    lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1
}

# Check prerequisites
echo "Checking prerequisites..."

if ! command_exists python3; then
    echo -e "${RED}❌ Python 3 is not installed${NC}"
    exit 1
fi

if ! command_exists node; then
    echo -e "${RED}❌ Node.js is not installed${NC}"
    exit 1
fi

if ! command_exists npm; then
    echo -e "${RED}❌ npm is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}✅ All prerequisites found${NC}"
echo ""

# Check if ports are available
echo "Checking port availability..."

if port_in_use 8000; then
    echo -e "${YELLOW}⚠️  Port 8000 is already in use (backend)${NC}"
    echo "Please stop the process using port 8000 or change the backend port"
    exit 1
fi

if port_in_use 3000; then
    echo -e "${YELLOW}⚠️  Port 3000 is already in use (frontend)${NC}"
    echo "Please stop the process using port 3000 or change the frontend port"
    exit 1
fi

echo -e "${GREEN}✅ Ports 8000 and 3000 are available${NC}"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down services..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "Services stopped."
    exit 0
}

# Set trap for cleanup
trap cleanup INT TERM

# Start backend server
echo "Starting AGUI Backend Server..."
cd agent
python3 agent.py &
BACKEND_PID=$!
cd ..

# Wait for backend to start
echo "Waiting for backend to initialize..."
sleep 5

# Check if backend is running
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Backend might not be fully initialized yet${NC}"
    echo "Continuing with frontend startup..."
fi

# Start frontend
echo ""
echo "Starting Frontend Application..."
npm run dev &
FRONTEND_PID=$!

# Wait for frontend to start
echo "Waiting for frontend to initialize..."
sleep 5

# Display status
echo ""
echo "============================================"
echo -e "${GREEN}AGUI RAG System Started Successfully!${NC}"
echo "============================================"
echo ""
echo "🚀 Backend (AGUI Server): http://localhost:8000"
echo "🎨 Frontend (CopilotKit): http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Keep script running
while true; do
    sleep 1
done