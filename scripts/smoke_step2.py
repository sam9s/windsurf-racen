from __future__ import annotations

import sys
from pathlib import Path

# Ensure local 'src' is importable when running as a script
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def ensure_outputs() -> Path:
    out = Path("outputs/step2")
    out.mkdir(parents=True, exist_ok=True)
    return out


ess = ensure_outputs()


def load_input() -> str:
    # Prefer the real fetch output from step1; fallback to local
    example_md = Path("outputs/step1/example_com.md")
    local_md = Path("outputs/step1/local_test.md")
    if example_md.exists():
        return example_md.read_text(encoding="utf-8")
    if local_md.exists():
        return local_md.read_text(encoding="utf-8")
    raise FileNotFoundError(
        "Run scripts/smoke_step1.py first to generate Markdown inputs"
    )


def run():
    from racen.log import get_logger
    from racen.step2_ingest import Cleaner, Chunker

    logger = get_logger("racen.smoke2")
    md = load_input()
    logger.info(f"Loaded input md ({len(md)} chars)")

    cleaner = Cleaner()
    cleaned = cleaner.clean(md)
    logger.info(f"Cleaned md ({len(cleaned)} chars)")

    chunker = Chunker(max_tokens=200, overlap_tokens=20)
    chunks = chunker.chunk(cleaned)
    logger.info(f"Chunks: {len(chunks)}")

    outdir = ensure_outputs()
    for ch in chunks:
        (outdir / f"{ch.id}.md").write_text(ch.text, encoding="utf-8")
    (outdir / "cleaned.md").write_text(cleaned, encoding="utf-8")
    logger.info(f"Wrote chunks to {outdir}")


if __name__ == "__main__":
    from racen.log import get_logger

    logger = get_logger("racen.smoke2")
    logger.info("Smoke Step 2: start")
    run()
    logger.info("Smoke Step 2: done")
