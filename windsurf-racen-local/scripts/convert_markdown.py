import argparse
from pathlib import Path
from urllib.parse import urlparse
import re

from markitdown import MarkItDown


def safe_name(url: str) -> str:
    p = urlparse(url)
    stem = (p.netloc + p.path).strip("/")
    if not stem:
        stem = p.netloc or "root"
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", stem)
    if len(stem) > 140:
        stem = stem[:140]
    return stem or "index"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("urls_file", help="Path to urls.txt from crawl step")
    parser.add_argument("--outdir", default="outputs/markdown", help="Output dir")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    crawl_md_dir = Path("outputs/markdown_crawl")

    md = MarkItDown(enable_plugins=False)

    count = 0
    with open(args.urls_file, "r", encoding="utf-8") as f:
        for line in f:
            url = line.strip()
            if not url:
                continue
            # Skip if Crawl4AI already produced markdown for this URL
            fname = safe_name(url) + ".md"
            if (crawl_md_dir / fname).exists():
                continue
            try:
                res = md.convert_uri(url)
                (outdir / fname).write_text(res.markdown or "", encoding="utf-8")
                count += 1
            except Exception as e:
                # If HTTPS fails, retry with HTTP for TLS-quirky sites
                if url.lower().startswith("https://"):
                    http_url = "http://" + url.split("://", 1)[1]
                    try:
                        res = md.convert_uri(http_url)
                        (outdir / fname).write_text(res.markdown or "", encoding="utf-8")
                        count += 1
                        continue
                    except Exception as e2:
                        print(f"[WARN] Failed to convert {url} and {http_url}: {e2}")
                else:
                    print(f"[WARN] Failed to convert {url}: {e}")

    print(f"Converted {count} pages to Markdown in {outdir}")


if __name__ == "__main__":
    main()
