FROM python:3.10-slim-bullseye
WORKDIR /usr/src/app

# Install sqlite3 and curl
RUN apt-get update && apt-get install -y  curl wget && rm -rf /var/lib/apt/lists/*

# Create a non-root user and setup the home directory
RUN groupadd -r appuser && useradd -m -d /home/appuser -r -g appuser appuser \
    && mkdir -p /data \
    && chown -R appuser:appuser /usr/src/app /data

# Copy the entire directory
COPY . .

RUN ls



# Switch to non-root user
USER appuser

# Adjust the PATH environment variable so that pip-installed programs are available
ENV PATH=/home/appuser/.local/bin:$PATH

# Set PYTHONPATH to include the directory where pip installs user packages
ENV PYTHONPATH=/home/appuser/.local/lib/python3.10/site-packages

# Installing requirements
# This is done as appuser, and we ensure the .local directory is used for the user's packages
RUN pip install --user -r requirements.txt

VOLUME /data
EXPOSE 8501




# Run app.py when the container launches
CMD ["streamlit", "run", "app.py"]