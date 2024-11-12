#!/bin/bash

# Default settings for FastAPI
DEFAULT_MODULE_NAME="src.backend.main"
MODULE_NAME=${MODULE_NAME:-$DEFAULT_MODULE_NAME}
VARIABLE_NAME=${VARIABLE_NAME:-app}
export APP_MODULE=${APP_MODULE:-"$MODULE_NAME:$VARIABLE_NAME"}

# Host settings
HOST=${HOST:-0.0.0.0}

# Port settings
FASTAPI_PORT=${FASTAPI_PORT:-8000}
STREAMLIT_PORT=${STREAMLIT_PORT:-8501}

echo "Starting MultimodalRAG in development mode..."

# Start FastAPI in the background
uvicorn --reload --proxy-headers --host $HOST --port $FASTAPI_PORT "$APP_MODULE" &
FASTAPI_PID=$!

# Start Streamlit
streamlit run src/frontend/Home.py \
    --server.address $HOST \
    --server.port $STREAMLIT_PORT \
    --browser.serverAddress "localhost" \
    --server.runOnSave true &
STREAMLIT_PID=$!

# Handle shutdown gracefully
trap 'kill $FASTAPI_PID $STREAMLIT_PID' SIGTERM SIGINT

# Wait for both processes
wait $FASTAPI_PID $STREAMLIT_PID