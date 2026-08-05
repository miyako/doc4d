#!/bin/sh
set -e

MODEL_URL="https://huggingface.co/keisuke-miyako/doc4d-2026-08-05/resolve/main/LFM2.5-Embedding-350M-Q8_0.gguf"
DATA_URL="https://huggingface.co/datasets/keisuke-miyako/doc4d-2026-08-05/resolve/main/doc.db"

mkdir -p models data

if [ ! -f models/LFM2.5-Embedding-350M-Q8_0.gguf ]; then
    echo "Downloading model from HF..."
    curl -L --retry 3 -o models/LFM2.5-Embedding-350M-Q8_0.gguf.tmp "$MODEL_URL"
    mv models/LFM2.5-Embedding-350M-Q8_0.gguf.tmp models/LFM2.5-Embedding-350M-Q8_0.gguf
fi

if [ ! -f data/doc.db ]; then
    echo "Downloading doc.db from HF..."
    curl -L --retry 3 -o data/doc.db.tmp "$DATA_URL"
    mv data/doc.db.tmp data/doc.db
fi

# Railway (and similar PaaS) assign a dynamic port via $PORT.
# Fall back to 80 for local/Docker/Oracle runs where PORT isn't set.
export PORT="${PORT:-80}"
export OMP_NUM_THREADS=1
envsubst '${PORT}' < /etc/nginx/nginx.conf.template > /etc/nginx/nginx.conf

nginx -t                # fail loudly here if the config is broken
nginx

# give nginx a moment to bind, then confirm it's actually up
sleep 1
if ! pgrep -x nginx > /dev/null; then
    echo "ERROR: nginx failed to start" >&2
    exit 1
fi
echo "nginx started OK, proxying :80 -> 127.0.0.1:7860"

exec python server.py
