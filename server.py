import sqlite3
import struct
import time
import llama_cpp
from llama_cpp import Llama
from typing import Optional
from typing import Literal
from mcp.server import MCPServer

DIM = 1024
DB_PATH = "data/doc.db"
MODEL_PATH = "models/LFM2.5-Embedding-350M-Q8_0.gguf"

QUERY_PREFIX = "query: "  # note trailing space, per model card
POOLING_TYPE = llama_cpp.LLAMA_POOLING_TYPE_CLS

_t0 = time.time()
llm = Llama(
    model_path=MODEL_PATH,
    embedding=True,
    pooling_type=POOLING_TYPE,
    n_ctx=512,
    n_threads=2,       # match your CPU allocation on the Space
    verbose=False,
)
print(f"[startup] model loaded in {time.time() - _t0:.2f}s", flush=True)

db = sqlite3.connect(DB_PATH, check_same_thread=False)
db.enable_load_extension(True)
import sqlite_vec
sqlite_vec.load(db)
db.enable_load_extension(False)

def embed_query(text: str) -> bytes:
    vec = llm.create_embedding(QUERY_PREFIX + text)["data"][0]["embedding"]
    if len(vec) != DIM:
        raise ValueError(f"Model returned {len(vec)}-dim vector, expected {DIM}")
    return struct.pack(f"{DIM}f", *vec)

mcp = MCPServer("rag-search")

@mcp.tool()
def search(
    query: str,
    language: Literal["en", "fr", "es", "pt", "ja"] = "en",
    version: Literal["18", "20", "21", "21-R3", "21-R4"] = "21-R4",
    full_text: bool = False,
    k: int = 10,
) -> list[dict]:
    """Search the corpus for passages semantically similar to the query."""
    k = max(1, min(k, 50))              # cap result count regardless of what caller requests
    query = query[:2000]                # cap query length fed into the embedder

    t0 = time.time()
    q_blob = embed_query(query)
    t1 = time.time()

    # pull extra candidates since we'll filter afterward
    candidate_limit = k * 20

    rows = db.execute("""
        SELECT url, text, language, version, distance
        FROM chunks
        WHERE embedding MATCH ?
        ORDER BY distance
        LIMIT ?
    """, (q_blob, candidate_limit)).fetchall()
    t2 = time.time()

    filtered = [
        r for r in rows
        if r[2] == language and r[3] == version
    ][:k]

    print(
        f"[search] embed={t1 - t0:.2f}s db={t2 - t1:.2f}s "
        f"total={t2 - t0:.2f}s candidates={len(rows)} returned={len(filtered)}",
        flush=True,
    )

    return [
        {
            "url": url,
            "similarity": 1 - distance,
            **({"text": text} if full_text else {}),
        }
        for url, text, lang, ver, distance in filtered
    ]

if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=7860)
