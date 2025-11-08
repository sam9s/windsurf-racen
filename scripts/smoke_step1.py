from __future__ import annotations

import pathlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def ensure_outputs() -> pathlib.Path:
    out = pathlib.Path("outputs/step1")
    out.mkdir(parents=True, exist_ok=True)
    return out


def smoke_url():
    from racen.step1_fetch_convert import MarkItDownClient
    from racen.log import get_logger

    logger = get_logger("racen.smoke")
    client = MarkItDownClient()
    url = "https://example.com/"
    md = client.convert_to_markdown(url=url)
    outdir = ensure_outputs()
    outfile = outdir / "example_com.md"
    outfile.write_text(md, encoding="utf-8")
    logger.info(f"Wrote: {outfile} ({len(md)} chars)")


def smoke_local_file():
    from racen.step1_fetch_convert import MarkItDownClient
    from racen.log import get_logger

    logger = get_logger("racen.smoke")
    html = """
    <html><head><title>Test</title></head>
    <body>
      <h1>Hello</h1>
      <p>RACEN smoke test paragraph.</p>
    </body></html>
    """.strip()

    tmp = pathlib.Path("outputs/step1/local_test.html")
    tmp.write_text(html, encoding="utf-8")

    client = MarkItDownClient()
    md = client.convert_to_markdown(file_path=str(tmp))

    outdir = ensure_outputs()
    outfile = outdir / "local_test.md"
    outfile.write_text(md, encoding="utf-8")
    logger.info(f"Wrote: {outfile} ({len(md)} chars)")


if __name__ == "__main__":
    from racen.log import get_logger

    logger = get_logger("racen.smoke")
    logger.info("Smoke Step 1: start")
    smoke_url()
    smoke_local_file()
    logger.info("Smoke Step 1: done")
