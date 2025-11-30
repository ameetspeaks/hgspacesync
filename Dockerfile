# UPGRADE TO PYTHON 3.10 (Fixes Google AI & ImportLib errors)
FROM python:3.10-slim

# Set working directory
WORKDIR /code

# Install system dependencies (needed for math libraries)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Create ephemeris folder and grant permissions
RUN mkdir -p /code/ephe && chmod 777 /code/ephe

# Copy the rest of the app code
COPY . .

# Start the app on port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]