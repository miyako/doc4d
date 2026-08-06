import sqlite3
import struct
import time
from typing import Literal

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from mcp.server import MCPServer

DIM = 1024
DB_PATH = "data/doc.db"
MODEL_PATH = "models/LFM2.5-Embedding-350M/model.onnx"
TOKENIZER_PATH = "models/LFM2.5-Embedding-350M/tokenizer.json"

QUERY_PREFIX = "query: "  # note trailing space, per model card

# NOTE: this bypasses fastembed's TextEmbedding class entirely. fastembed only
# accepts models from its own built-in registry (BGE, e5, MiniLM, etc.) — it
# has no way to load an arbitrary local ONNX file like your converted LFM2.5
# checkpoint. Loading it directly via onnxruntime + tokenizers is the correct
# approach for a custom/non-registry model.
#
# UNVERIFIED: input/output tensor names and pooling below assume a standard
# BERT-style encoder export (input_ids/attention_mask in, last_hidden_state
# out, CLS-token pooling to match your original POOLING_TYPE_CLS). If your
# actual model.onnx uses different input names or pooling, this will need
# adjusting once you see the real error message.

_t0 = time.time()
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
_input_names = {i.name for i in session.get_inputs()}
print(f"[startup] model loaded in {time.time() - _t0:.2f}s, inputs={_input_names}", flush=True)

db = sqlite3.connect(DB_PATH, check_same_thread=False)
db.enable_load_extension(True)
import sqlite_vec
sqlite_vec.load(db)
db.enable_load_extension(False)


def embed_query(text: str) -> bytes:
    encoding = tokenizer.encode(QUERY_PREFIX + text)
    input_ids = np.array([encoding.ids], dtype=np.int64)
    attention_mask = np.array([encoding.attention_mask], dtype=np.int64)

    onnx_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    if "token_type_ids" in _input_names:
        onnx_inputs["token_type_ids"] = np.zeros_like(input_ids)

    outputs = session.run(None, onnx_inputs)
    last_hidden_state = outputs[0]           # shape: (1, seq_len, hidden_dim)
    vec = last_hidden_state[0, 0, :]          # CLS token, matches original POOLING_TYPE_CLS

    # llama.cpp's create_embedding L2-normalizes by default; matching that here
    # so cosine-distance search against doc.db behaves the same as before.
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    if len(vec) != DIM:
        raise ValueError(f"Model returned {len(vec)}-dim vector, expected {DIM}")
    return struct.pack(f"{DIM}f", *vec.tolist())


mcp = MCPServer("4D Documentation")


@mcp.tool()
def search(
    query: str,
    language: Literal["en", "fr", "es", "pt", "ja"] = "en",
    version: Literal["18", "20", "21", "21-R3", "21-R4"] = "21-R4",
    full_text: bool = True,
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
