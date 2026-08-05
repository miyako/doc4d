FROM python:3.12-slim

RUN apt-get update && apt-get install -y build-essential cmake git nginx && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .
COPY models/ ./models/
COPY data/ ./data/

COPY nginx.conf /etc/nginx/nginx.conf
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# 7860 stays internal-only now; 80 is what actually gets published/exposed
EXPOSE 80

ENTRYPOINT ["/entrypoint.sh"]
