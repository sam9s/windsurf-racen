import asyncio
from pathlib import Path
from typing import List
from urllib.parse import urlparse
import re

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

DEFAULT_URLS: List[str] = [
    "https://grest.in/pages/faqs",
    "https://grest.in/policies/refund-policy",
    "https://grest.in/policies/shipping-policy",
    "https://grest.in/policies/terms-of-service",
    "https://grest.in/pages/warranty",
]


def safe_name(url: str) -> str:
    p = urlparse(url)
    stem = (p.netloc + p.path).strip("/")
    if not stem:
        stem = p.netloc or "root"
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", stem)
    if len(stem) > 140:
        stem = stem[:140]
    return stem or "index"


async def fetch_and_save(urls: List[str]) -> int:
    outdir = Path("outputs/markdown_crawl")
    outdir.mkdir(parents=True, exist_ok=True)

    browser = BrowserConfig(headless=True, verbose=False)
    run = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)

    saved = 0
    async with AsyncWebCrawler(config=browser) as crawler:
        for u in urls:
            print(f"[C4AI] Fetching: {u}", flush=True)
        results = await crawler.arun_many(urls=urls, config=run)
        for res in results:
            if getattr(res, "success", False) and getattr(res, "markdown", None):
                fname = safe_name(res.url) + ".md"
                (outdir / fname).write_text(res.markdown or "", encoding="utf-8")
                saved += 1
                print(f"[C4AI] Saved: {res.url} -> {fname}", flush=True)
            else:
                print(
                    f"[WARN] Failed: {res.url} -> {getattr(res, 'error_message', 'no markdown')} ",
                    flush=True,
                )
    return saved


def main() -> None:
    urls = DEFAULT_URLS
    saved = asyncio.run(fetch_and_save(urls))
    print(f"Saved markdown for {saved}/{len(urls)} known URLs into outputs/markdown_crawl/")


if __name__ == "__main__":
    main()
