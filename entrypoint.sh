#!/bin/sh
set -e

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
