import time, glob, os, traceback
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker

BASE = r"windsurf-racen-local\outputs\markdown_crawl"
files = sorted(glob.glob(os.path.join(BASE, "*.md")))
if not files:
    raise SystemExit(f"No markdown files found in: {os.path.abspath(BASE)}")

f = files[0]
print("smoke_file:", os.path.basename(f))

t0 = time.time()
try:
    conv = DocumentConverter().convert(source=f)
    dl_doc = conv.document
    print("has_document:", dl_doc is not None)

    # Some docling builds want positional arg (no keyword).
    chunker = HybridChunker()
    try:
        chunks = list(chunker.chunk(dl_doc=dl_doc))
    except TypeError:
        # Fallback for signature mismatch
        chunks = list(chunker.chunk(dl_doc))

    print("chunks_count:", len(chunks))
    if chunks:
        print("first_chunk_100:", chunks[0].text[:100].replace("\n", " "))

except Exception:
    print("!! Exception during conversion/chunking:")
    traceback.print_exc()
finally:
    print("elapsed_sec:", round(time.time() - t0, 2))
