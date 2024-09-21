# Use Docker BuildKit for better performance
# Enable BuildKit by setting the environment variable before building:
# export DOCKER_BUILDKIT=1

# Base image with Python 3.11
FROM python:3.11-slim-buster

# Install system dependencies and clean up in a single RUN command
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    build-essential \
    curl \
    gcc \
    git \
    libgl1 \
    libmagic-dev \
    libpq-dev \
    poppler-utils \
    && apt-get clean \
    && rm -rf /var/cache/apt/* \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables to optimize Python and pip
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_ROOT_USER_ACTION=ignore

# Set the working directory
WORKDIR /src

# Copy only requirements.txt first to leverage Docker cache
COPY requirements.txt .

# Upgrade pip and install dependencies in a single RUN command
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Add /src/scripts to PATH
ENV PATH="$PATH:/src/scripts"

# Command to start the application
CMD ["./scripts/start-dev.sh"]
