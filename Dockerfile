# Base image with Python 3.11
FROM python:3.11-slim-buster

# Install system dependencies
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    build-essential \
    curl \
    gcc \
    git \
    libgl1 \
    libmagic-dev \
    libpq-dev \
    poppler-utils \
    && apt-get clean && \
    rm -rf /var/cache/apt/* && rm -rf /var/lib/apt/lists/*

# Environment variables to optimize Python and pip
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_ROOT_USER_ACTION=ignore

# Copy the requirements file
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the application code into the container
COPY . /src

# Add /src/scripts to PATH
ENV PATH "$PATH:/src/scripts"

# Set the working directory
WORKDIR /src

# Command to start the application
CMD ["./scripts/start-dev.sh"]
