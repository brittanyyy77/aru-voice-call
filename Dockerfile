FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uncomment to include local whisper support in the image:
# RUN pip install --no-cache-dir faster-whisper==1.1.0

COPY . .

EXPOSE 8443

CMD ["python", "-m", "server.main"]
