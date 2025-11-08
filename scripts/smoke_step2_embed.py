from __future__ import annotations

import sys
from pathlib import Path

# Ensure local 'src' is importable when running as a script
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racen.log import get_logger
from racen.step2_embed import OpenAIEmbedder

logger = get_logger("racen.smoke2.embed")


def main():
    # Load one chunk from step2 or fallback to step1 cleaned text
    step2_dir = Path("outputs/step2")
    if not step2_dir.exists():
        raise FileNotFoundError("Run scripts/smoke_step2.py first")

    chunk_files = sorted(step2_dir.glob("c*.md"))
    if not chunk_files:
        raise FileNotFoundError("No chunks found; run scripts/smoke_step2.py")

    text = chunk_files[0].read_text(encoding="utf-8")

    embedder = OpenAIEmbedder()
    res = embedder.embed(id=chunk_files[0].stem, text=text, metadata={"source": str(chunk_files[0])})

    out = step2_dir / f"{res.id}.json"
    out.write_text(
        "{\n" +
        f"  \"id\": \"{res.id}\",\n" +
        f"  \"model\": \"{res.model}\",\n" +
        f"  \"dim\": {res.dim},\n" +
        f"  \"vector_preview\": [{', '.join(str(round(v, 3)) for v in res.vector[:8])}]\n" +
        "}\n",
        encoding="utf-8",
    )
    logger.info(f"Wrote embedding json: {out}")


if __name__ == "__main__":
    logger.info("Smoke Step 2 (embed): start")
    main()
    logger.info("Smoke Step 2 (embed): done")
