import sqlite3
import struct
import time
import llama_cpp
from llama_cpp import Llama
from typing import Optional
from typing import Literal
from mcp.server import MCPServer

print(llama_cpp.llama_cpp.llama_print_system_info().decode())

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
    n_threads=1,       # testing: llama.cpp's multi-thread sync busy-spins,
                       # which can be catastrophically slow on throttled/shared
                       # vCPU containers — try single-threaded before assuming
                       # more cores/CPU tuning is the answer
    verbose=False,
    use_mmap=False,
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

mcp = MCPServer("4D Documentation")

@mcp.tool()
def search(
    query: str,
    language: Literal["en", "fr", "es", "pt", "ja"] = "en",
    version: Literal["18", "20", "21", "21-R3", "21-R4"] = "21-R4",
    full_text: bool = True,
    k: int = 10,
) -> list[dict]:
    r"""Search the official 4D programming language and database documentation.

    Use this tool for any technical question about 4D — syntax, commands,
    classes, ORDA, forms, project structure, deployment, or server
    administration — whether or not the user names "4D." Prefer this over
    general knowledge or web search for ambiguous technical queries (e.g.
    "how do I declare a class property") when 4D is the active context,
    since it returns passages from the current, versioned docs.
    
    Params: version defaults to "21-R4" unless the user names another
    ("18", "20", "21", "21-R3"). language matches the user's dominant
    conversation language ("en", "fr", "es", "pt", "ja"), not English by
    default. full_text=True (default) inlines the matched passage; set False
    only when you want a bare list of URLs without content.
    
    Return contract: each result is a documentation excerpt, not a fact you
    already knew — its `url` is that excerpt's source. When full_text=True,
    the `text` field is itself returned as "Source: <url>\n\n<passage>", so
    the url travels with the passage as part of the same string, not as a
    separate field you have to remember to check. When you use a passage's
    content in your answer, carry that passage's "Source:" line forward as
    the citation for the sentence it supports — an answer built from these
    results without the matching url next to each claim is missing part of
    the tool's output, not just missing a nice-to-have.
    
    ## Writing conventions
    The following conventions are used in the 4D language documentation:
    - the `{ }` characters (braces) indicate optional parameters. For example, `.delete({ option : Integer })` means that the *option* parameter may be omitted when calling the function.
    - the `any` keyword is used for parameters that can be a value of any type (number, text, boolean, date, time, object, collection...).
    - when a parameter can accept several types, they are listed and separated by comma, for example: `value : Text, Real, Date, Time`
    This means the parameter *value* can be Text OR Real OR Date OR Time.
    - **variadic parameter**: the `...param : Type` notation indicates from 0 to an unlimited number of parameters of the same type. For example, `.concat( value : any { ;...valueN : any }) : Collection` means that an unlimited number of values of any type can be passed to the function.
    - **variadic group of parameters**: the `{; ...(param1 : Type ; param2 : Type)}` notation indicates from 1 to an unlimited number of groups of parameters. For example, `COLLECTION TO ARRAY( collection : Collection ; array : Array {; propertyName : Text}{; ...(array : Array ; propertyName : Text) })` means that an unlimited number of couple values of type array/text can be passed to the command.
    ### Parameter type description
    In the 4D language documentation, the following parameter types can be used. 
    |Type | Definition | Examples of a 4D command using it|
    |-- | -- | --|
    |>, <, >=, <=, #, =, \| , % | Comparison, logical operators or symbols used in query conditions or expressions.| ORDER BY([Products];[Products]Type;<)<br/>PRINT RECORD([Employees];>)|
    |any | A parameter that can accept any supported data type | JSON Stringify($value)<br/>$col.push(6;New object("firstname";"John"))|
    |Array | A variable containing a list of values of the same type. | ARRAY TEXT($arr;10)|
    |BLOB array | An array containing BLOB values. | ARRAY BLOB($data;10)|
    |Blob | Binary large object used to store binary data. | BLOB TO DOCUMENT($blob;"file.bin")|
    |Boolean | A logical value: True or False. | If (OK=1)|
    |Boolean array | An array containing boolean values. | ARRAY BOOLEAN($flags;10)|
    |Class name (ex: 4D.File) | A reference to a class type used to create or manipulate class instances. | $file:=File("/RESOURCES/NovelCover1.jpg")|
    |Collection | An ordered list of values that can contain multiple types. | New collection("A";"B";"C")|
    |Date | A calendar date value. | $vDate:=Current date|
    |Date array | An array containing date values. | ARRAY DATE($dates;10)|
    |Expression| Can be anything |  SET PROCESS VARIABLE($vlProcess;vtCurStatus;"")|
    |Field | A reference to a field belonging to a table. | ORDER BY([Person];[Person]Name)|
    |Integer | A whole number without decimal part. |  $Sel:=ds.Employee.newSelection(dk keep ordered)|
    |Integer array | An array containing integer values. | ARRAY INTEGER($numbers;10)|
    |Longint array | An array containing long integer values. | ARRAY LONGINT($values;10)|
    |Object array | An array containing objects. | ARRAY OBJECT($objects;10)|
    |Object | A structured data container composed of key/value pairs. | $entity.fromObject($o)|
    |Operator | Always *. | QUERY([Person];[Person]Name="Smith";*)|
    |Picture array | An array containing pictures. | ARRAY PICTURE($images;10)|
    |Picture | A graphical image value. | READ PICTURE FILE($pic;"image.png")|
    |Pointer array | An array containing pointers. | ARRAY POINTER($ptrs;10)|
    |Pointer | A reference to another variable, field, or object. |  If(Is nil pointer($ptr))|
    |Real array | An array containing real numbers. | ARRAY REAL($values;10)|
    |Real | A floating-point numeric value. |  $vlResult:=Int(123.4)|
    |Table | A reference to a database table. | ALL RECORDS([Person])|
    |Text | A sequence of characters representing textual data. | ALERT("Hello world")|
    |Text array | An array containing text values. | ARRAY TEXT($names;10)|
    |Time | A time value representing hours, minutes, and seconds. | Current time|
    |Time array | An array containing time values. | ARRAY TIME($times;10)|
    |Variable | A writable variable of type "any" that can receive a value (assignable). | SET PICTURE METADATA(vPicture;IPTC keywords;$arrTkeywords)|
    """
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
            **({"text": f"Source: {url}\n\n{text}"} if full_text else {}),
        }
        for url, text, lang, ver, distance in filtered
    ]

if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=7860)
