from __future__ import annotations

import argparse
from html.parser import HTMLParser
from pathlib import Path
from typing import List, Set
from urllib.parse import urljoin, urlparse

import requests
import yaml  # type: ignore


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX_URL = "https://grest.in/blogs/news"
DEFAULT_OUT = ROOT / "Grest_Data" / "grest_blog_news.yaml"


class _LinkCollector(HTMLParser):
    """Collect all href values from anchor tags in an HTML document.

    This is a small helper around :class:`HTMLParser` so we can keep the
    extraction logic dependency-free (no BeautifulSoup required).
    """

    def __init__(self) -> None:
        """Initialize the collector with an empty href list."""

        super().__init__()
        self.hrefs: List[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Record href attributes from ``<a>`` tags.

        Args:
            tag (str): HTML tag name.
            attrs (list[tuple[str, str | None]]): Attribute name/value pairs.
        """

        if tag.lower() != "a":
            return
        for name, value in attrs:
            if name.lower() == "href" and value:
                self.hrefs.append(value)


def extract_blog_urls_from_html(html: str, index_url: str = DEFAULT_INDEX_URL) -> List[str]:
    """Extract blog article URLs under ``/blogs/news`` from HTML content.

    This helper is intentionally pure (no network calls) so that tests can
    exercise it with synthetic HTML fixtures.

    Args:
        html (str): HTML content of the blog index page.
        index_url (str): URL of the index page used as base for resolving
            relative links.

    Returns:
        List[str]: Sorted list of unique absolute article URLs that live under
        ``https://grest.in/blogs/news/...``.
    """

    parser = _LinkCollector()
    parser.feed(html)

    seen: Set[str] = set()
    out: List[str] = []

    for raw_href in parser.hrefs:
        if not raw_href:
            continue

        absolute = urljoin(index_url, raw_href)
        parsed = urlparse(absolute)

        # Only keep grest.in blog links.
        if parsed.netloc not in {"grest.in", "www.grest.in"}:
            continue
        if not parsed.path.startswith("/blogs/news"):
            continue

        # Skip the index page itself (/blogs/news or /blogs/news/).
        if parsed.path.rstrip("/") == "/blogs/news":
            continue

        # Canonicalize to scheme + host + path (drop query/fragment).
        canonical = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

        if canonical in seen:
            continue
        seen.add(canonical)
        out.append(canonical)

    out.sort()
    return out


def _fetch_html(url: str, timeout: int = 20) -> str:
    """Fetch raw HTML for a given URL.

    Args:
        url (str): Target URL.
        timeout (int): Request timeout in seconds.

    Returns:
        str: Response body as text.
    """

    resp = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "RACEN/0.1 (+https://grest.in)"},
    )
    resp.raise_for_status()
    return resp.text


def _write_yaml(urls: List[str], out_path: Path) -> None:
    """Write discovered blog URLs to a YAML file under the ``blogs`` key.

    Args:
        urls (List[str]): Absolute blog article URLs.
        out_path (Path): Output YAML path.
    """

    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = {"blogs": urls}
    with out_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)


def main() -> int:
    """Entry point for building a blog YAML from the Grest news index.

    This script fetches the index page (``/blogs/news``), discovers article
    links, and writes them to ``Grest_Data/grest_blog_news.yaml`` by default.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Discover Grest blog article URLs under /blogs/news and "
            "write them to a YAML file."
        )
    )
    parser.add_argument(
        "--index-url",
        type=str,
        default=DEFAULT_INDEX_URL,
        help="Blog index URL to scan (default: https://grest.in/blogs/news)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(DEFAULT_OUT),
        help=(
            "Path to output YAML file (default: Grest_Data/grest_blog_news.yaml)"
        ),
    )
    args = parser.parse_args()

    html = _fetch_html(args.index_url)
    urls = extract_blog_urls_from_html(html, index_url=args.index_url)

    _write_yaml(urls, Path(args.out))

    print(f"Discovered {len(urls)} blog URLs from {args.index_url}")
    print(f"Written to: {args.out}")

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
