FROM python:3.12-slim

RUN apt-get update && apt-get install -y build-essential cmake git nginx curl gettext-base && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# llama-cpp-python's prebuilt wheel targets a generic/lowest-common-denominator
# CPU, which can be slow. GGML_NATIVE auto-detection is unreliable on virtualized
# cloud CPUs (cpuid restrictions can make it think there's no SIMD support at
# all, producing a much SLOWER scalar-only build). Force explicit, safe flags
# that virtually all modern server CPUs support instead.
ENV CMAKE_ARGS="-DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_NATIVE=OFF"
RUN pip install --no-cache-dir --force-reinstall --no-binary llama-cpp-python llama-cpp-python

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .
# models/ and data/ are no longer copied at build time — entrypoint.sh
# downloads LFM2.5-Embedding-350M-Q8_0.gguf and doc.db from Hugging Face
# at container startup instead, keeping this image lightweight.

COPY nginx.conf.template /etc/nginx/nginx.conf.template
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# 7860 stays internal-only. Public port is dynamic: Railway injects $PORT at
# runtime; entrypoint.sh falls back to 80 when $PORT isn't set (e.g. Oracle/local).
EXPOSE 80

ENTRYPOINT ["/entrypoint.sh"]
