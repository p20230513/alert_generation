FROM python:3.9-slim

WORKDIR /app

# Install system dependencies if any
# RUN apt-get update && apt-get install -y ...

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
