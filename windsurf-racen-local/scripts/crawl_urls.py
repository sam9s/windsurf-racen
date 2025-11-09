import asyncio
import argparse
from pathlib import Path
from urllib.parse import urldefrag, urlparse
from typing import Set, List
from urllib.parse import urlparse as _urlparse
import re

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    CacheMode,
    MemoryAdaptiveDispatcher,
)


def normalize_url(url: str) -> str:
    return urldefrag(url)[0]


def same_domain(start: str, candidate: str) -> bool:
    s = urlparse(start)
    c = urlparse(candidate)
    return (s.scheme in ("http", "https")) and (s.netloc == c.netloc)


async def crawl_recursive(start_urls: List[str], max_depth: int, max_concurrent: int) -> Set[str]:
    browser_config = BrowserConfig(headless=True, verbose=False)
    run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent,
    )

    visited: Set[str] = set()
    current_urls: Set[str] = {normalize_url(u) for u in start_urls}
    origin = start_urls[0]

    async with AsyncWebCrawler(config=browser_config) as crawler:
        for _ in range(max_depth):
            to_crawl = [u for u in current_urls if u not in visited]
            if not to_crawl:
                break
            results = await crawler.arun_many(
                urls=to_crawl,
                config=run_config,
                dispatcher=dispatcher,
            )
            next_level: Set[str] = set()
            for result in results:
                visited.add(result.url)
                # Save markdown if available
                if getattr(result, "success", False) and getattr(result, "markdown", None):
                    try:
                        outdir = Path("outputs/markdown_crawl")
                        outdir.mkdir(parents=True, exist_ok=True)
                        fname = safe_name(result.url) + ".md"
                        (outdir / fname).write_text(result.markdown or "", encoding="utf-8")
                    except Exception:
                        pass
                if result.success and result.links:
                    for link in result.links.get("internal", []):
                        href = normalize_url(link.get("href", ""))
                        if href and same_domain(origin, href) and href not in visited:
                            next_level.add(href)
            current_urls = next_level

    return visited


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="Start URL, e.g., https://grest.in/")
    parser.add_argument("--depth", type=int, default=2, help="Max crawl depth (hops)")
    parser.add_argument("--concurrency", type=int, default=10, help="Parallel fetches")
    parser.add_argument("--outdir", default="outputs", help="Output directory")
    args = parser.parse_args()

    urls = asyncio.run(
        crawl_recursive([args.start], max_depth=args.depth, max_concurrent=args.concurrency)
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    txt_path = outdir / "urls.txt"
    jsonl_path = outdir / "urls.jsonl"

    with txt_path.open("w", encoding="utf-8") as f:
        for u in sorted(urls):
            f.write(f"{u}\n")

    with jsonl_path.open("w", encoding="utf-8") as f:
        for u in sorted(urls):
            f.write(f"{{\"url\": \"{u}\"}}\n")

    print(f"Discovered {len(urls)} URLs")
    print(f"Wrote: {txt_path} and {jsonl_path}")


def safe_name(url: str) -> str:
    p = _urlparse(url)
    stem = (p.netloc + p.path).strip("/")
    if not stem:
        stem = p.netloc or "root"
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", stem)
    if len(stem) > 140:
        stem = stem[:140]
    return stem or "index"


if __name__ == "__main__":
    main()
