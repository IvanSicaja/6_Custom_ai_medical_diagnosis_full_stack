# Use the official slim variant of the Python image with version 3.11 as the base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY p3.11.9requirements_DOCKER.txt .

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install the dependencies
RUN pip install tensorflow

# Install the dependencies
RUN pip install --no-cache-dir -r p3.11.9requirements_DOCKER.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port on which your application will run (adjust as needed)
EXPOSE 5000

# Define the command to run your application
CMD ["python", "main.py"]
