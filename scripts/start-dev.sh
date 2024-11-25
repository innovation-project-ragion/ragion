#!/bin/bash

# Default application module and variables
DEFAULT_MODULE_NAME="src.main"

MODULE_NAME=${MODULE_NAME:-$DEFAULT_MODULE_NAME}
VARIABLE_NAME=${VARIABLE_NAME:-app}
export APP_MODULE=${APP_MODULE:-"$MODULE_NAME:$VARIABLE_NAME"}

HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}

echo "Starting MultimodalRAG in development mode..."

# Start the main application in the background
echo "Launching Uvicorn server for $APP_MODULE on $HOST:$PORT..."
uvicorn --reload --proxy-headers --host $HOST --port $PORT "$APP_MODULE" &

# Allow the application to start before exposing services
echo "Waiting for application startup..."
sleep 10

# Run the expose_services.py script
echo "Exposing services via Localtunnel..."
python /src/expose_services.py

# Keep the container alive
tail -f /dev/null
