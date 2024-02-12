# Use an official Python runtime as a parent image
FROM ubuntu:latest

# Set the working directory in the container
WORKDIR /app

# Install build dependencies and Python
RUN apt-get update && \
    apt-get install -y gcc python3-pip python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and setuptools
RUN pip install --upgrade pip setuptools

# Copy only the specified directories and files into the container at /app
COPY src /app/src
COPY static /app/static
COPY app.py /app/app.py
COPY Dockerfile /app/Dockerfile
COPY requirements-docker.txt /app/requirements.txt

# Install dependencies from requirements.txt
RUN apt-get update && apt-get install -y ffmpeg \
    && pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run app.py when the container launches
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
