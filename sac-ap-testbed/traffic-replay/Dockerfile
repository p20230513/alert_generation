FROM python:3.9-slim

WORKDIR /app

# Install system dependencies that scapy might need
RUN apt-get update && apt-get install -y tcpdump

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# The container will just idle until the defender executes a script inside it.
CMD ["tail", "-f", "/dev/null"]
