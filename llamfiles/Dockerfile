FROM ubuntu:latest
WORKDIR /usr/src/app

# Install sqlite3 and curl
RUN apt-get update && apt-get install -y curl wget && rm -rf /var/lib/apt/lists/*

# Create a non-root user and setup the home directory
RUN groupadd -r appuser && useradd -m -d /home/appuser -r -g appuser appuser \
    && mkdir -p /data \
    && chown -R appuser:appuser /usr/src/app /data

# Switch to non-root user
USER appuser

# Adjust the PATH environment variable so that pip-installed programs are available
ENV PATH=/home/appuser/.local/bin:$PATH

# Set PYTHONPATH to include the directory where pip installs user packages
ENV PYTHONPATH=/home/appuser/.local/lib/python3.10/site-packages

EXPOSE 8081



# Run app.py when the container launches
CMD ["bash","./mistral-7b-instruct-v0.1-Q4_K_M-server.llamafile", "--port", "8081", "--host", "0.0.0.0","-ngl", "5"]
