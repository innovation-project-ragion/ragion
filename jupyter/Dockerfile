# jupyter/Dockerfile

# Use the official Jupyter Data Science Notebook as the base image
FROM jupyter/datascience-notebook:latest

# Switch to root to install packages
USER root

# Copy the requirements.txt from the build context (project root) into the container
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Switch back to the default jovyan user
USER $NB_UID

# Expose the default Jupyter port
EXPOSE 8888

# (Optional) Start Jupyter Lab automatically (handled by base image)
# CMD ["start.sh", "jupyter", "lab"]
