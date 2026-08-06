FROM python:3.12-slim

# build-essential/cmake/git were only needed to compile llama-cpp-python from
# source. fastembed's ONNX Runtime backend ships as a prebuilt wheel, so no
# compiler toolchain is required anymore.
RUN apt-get update && apt-get install -y nginx curl gettext-base procps && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .
# models/ and data/ are no longer copied at build time — entrypoint.sh
# downloads the ONNX model and doc.db from Hugging Face at container
# startup instead, keeping this image lightweight.

COPY nginx.conf.template /etc/nginx/nginx.conf.template
COPY icons/ /usr/share/nginx/html/icons/
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# 7860 stays internal-only. Public port is dynamic: Railway injects $PORT at
# runtime; entrypoint.sh falls back to 80 when $PORT isn't set (e.g. Oracle/local).
EXPOSE 80

ENTRYPOINT ["/entrypoint.sh"]
