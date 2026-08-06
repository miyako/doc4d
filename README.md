# doc4d

MCP server for semantic search over the [4D Documentation](https://developer.4d.com/) corpus.

It exposes a single MCP tool, `search`, which embeds a natural-language query and returns the most semantically similar documentation passages (URL + text) for a given language and 4D product version.

A live demo is running at:

```
https://doc4d-production.up.railway.app
```

(streamable-http MCP transport, mounted at `/`)

## How it works

- **Corpus & vector index:** [`keisuke-miyako/doc4d-2026-08-05`](https://huggingface.co/datasets/keisuke-miyako/doc4d-2026-08-05) on Hugging Face — a SQLite database (`doc.db`) with a [`sqlite-vec`](https://github.com/asg017/sqlite-vec) `vec0` virtual table of 1024-dim embeddings, chunked text, and `url` / `language` / `version` metadata for each chunk of 4D documentation.
- **Embedding model:** [`LFM2.5-Embedding-350M`](https://huggingface.co/LiquidAI/LFM2.5-Embedding-350M-GGUF), exported to ONNX and run locally via `onnxruntime` + `tokenizers` (no external embedding API call per query).
- **Server:** `server.py` loads the ONNX model and the SQLite DB at startup, embeds incoming queries with a CLS-pooled, L2-normalized vector (matching the pooling used to build the original embeddings), and runs a cosine-distance nearest-neighbor search via `sqlite-vec`'s `MATCH` operator.
- **Transport:** MCP over `streamable-http`, served internally on port `7860` and reverse-proxied by nginx, which also handles CORS and basic per-IP rate limiting.

### `search` tool

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `query` | `string` | — | Free-text query, truncated to 2000 chars. |
| `language` | `"en" \| "fr" \| "es" \| "pt" \| "ja"` | `"en"` | Filters results to this language. |
| `version` | `"18" \| "20" \| "21" \| "21-R3" \| "21-R4"` | `"21-R4"` | Filters results to this 4D product version. |
| `full_text` | `bool` | `true` | If `false`, omits the `text` field from results (URL + similarity only). |
| `k` | `int` | `10` | Number of results to return, capped at 50. |

Returns a list of `{ url, similarity, text? }` objects, ordered by descending similarity.

Under the hood, `k * 20` nearest-neighbor candidates are pulled from `sqlite-vec` and then filtered down to the requested `language`/`version`, since the vector index itself isn't partitioned by those fields.

## Project layout

```
.
├── Dockerfile
├── entrypoint.sh          # downloads model + DB from HF, starts nginx, then server.py
├── nginx_conf.template    # reverse proxy, CORS, rate limiting
├── requirements.txt
├── server.py              # MCP server + embedding + search logic
└── LICENSE
```

The Docker image ships **without** the model or database baked in — `entrypoint.sh` downloads them from Hugging Face on container start:

- `models/LFM2.5-Embedding-350M/model.onnx` and `tokenizer.json` from `keisuke-miyako/doc4d-2026-08-05` (model repo)
- `data/doc.db` from [`datasets/keisuke-miyako/doc4d-2026-08-05`](https://huggingface.co/datasets/keisuke-miyako/doc4d-2026-08-05)

This keeps the image small and lets the corpus/model be updated without rebuilding the image — just clear the mounted volume (or redeploy) to force a re-download.

## Running locally

**Docker (recommended):**

```bash
docker build -t doc4d .
docker run --rm -p 8080:80 -e PORT=80 doc4d
```

The server will download the model and database on first start (this can take a minute depending on connection speed), then listen on `http://localhost:8080`.

**Without Docker:**

```bash
pip install -r requirements.txt
# also requires nginx if you want the proxy/CORS/rate-limiting layer,
# or point an MCP client directly at 127.0.0.1:7860 and skip nginx
./entrypoint.sh
```

`entrypoint.sh` expects to be run from the repo root and will create `models/` and `data/` alongside it.

## Deploying to Railway

This repo is set up to deploy on [Railway](https://railway.com) with zero config beyond the Dockerfile:

1. **New Project → Deploy from GitHub repo**, select `miyako/doc4d`.
2. Railway detects the `Dockerfile` automatically and builds it — no build command needed.
3. Railway injects `$PORT` at runtime; `entrypoint.sh` picks it up automatically and templates it into the nginx config (`envsubst '${PORT}'`), so **no manual port configuration is required**.
4. First boot will take longer than subsequent restarts, since `entrypoint.sh` downloads `model.onnx`, `tokenizer.json`, and `doc.db` from Hugging Face before starting the server. If you want faster cold starts, attach a [Railway volume](https://docs.railway.com/reference/volumes) mounted at `/app/models` and `/app/data` so those files persist across deploys/restarts instead of being re-downloaded every time.
5. Once deployed, Railway gives you a public URL (e.g. `https://<your-app>.up.railway.app`) — that's your MCP `streamable-http` endpoint.

No environment variables are required for a default deploy. `PORT` is set by Railway automatically; `OMP_NUM_THREADS=1` is set internally by `entrypoint.sh` to keep ONNX Runtime CPU usage predictable on shared/small instances.

### Deploying elsewhere (Oracle Cloud, bare Docker host, etc.)

The same image works anywhere that can run a container and reach Hugging Face over HTTPS:

- If `$PORT` isn't set, `entrypoint.sh` falls back to port `80`.
- Make sure outbound HTTPS to `huggingface.co` is allowed on first boot (for the model/DB download).
- Persist `models/` and `data/` on a volume if you want to avoid re-downloading ~350M-model-sized assets on every restart.

## Connecting an MCP client

Point any MCP client that supports `streamable-http` transport at the server's base URL, e.g. for the hosted demo:

```
https://doc4d-production.up.railway.app
```

Rate limiting (5 req/s per IP, burst 10) and CORS (`Access-Control-Allow-Origin: *`) are applied at the nginx layer in front of the MCP server.

## Notes / caveats

- The embedding pooling and normalization in `server.py` assume a standard BERT-style ONNX export (CLS-token pooling, L2-normalized output) matching the original `llama.cpp`-based embedding generation used to build the dataset. If you swap in a different ONNX export of the model, double check `embed_query()` still matches.
- Query results are only as fresh as the `doc4d-2026-08-05` dataset snapshot — see the [dataset card](https://huggingface.co/datasets/keisuke-miyako/doc4d-2026-08-05) for details on how it was built and its limitations.
- The `language`/`version` filter happens **after** vector search on an over-fetched candidate set (`k * 20`), not natively in the index — if you query for a rare `language`/`version` combination, you may get fewer than `k` results even when more exist in the corpus.

## License

MIT — see [LICENSE](./LICENSE).
