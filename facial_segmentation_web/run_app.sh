#!/bin/bash

echo "Starting Facial Segmentation Web Application..."

# Function to cleanup on exit
cleanup() {
    echo "Stopping all services..."
    pkill -f "python app.py" || true
    pkill -f "node" || true
    echo "Application stopped."
    exit 0
}

# Setup trap to catch Ctrl+C
trap cleanup SIGINT

# Start backend
echo "Starting backend server..."
cd backend
./run.sh &
backend_pid=$!

# Wait for backend to initialize
echo "Waiting for backend to initialize..."
sleep 5

# Start frontend
echo "Starting frontend application..."
cd ../frontend
./run.sh &
frontend_pid=$!

echo "Application started! Frontend will be available at http://localhost:3000"
echo "Press Ctrl+C to stop all services..."

# Wait for processes to complete
wait $backend_pid $frontend_pid 