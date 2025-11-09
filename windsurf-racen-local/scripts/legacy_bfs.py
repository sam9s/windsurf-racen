import argparse
from collections import deque
from pathlib import Path
from urllib.parse import urlparse, urljoin, urldefrag
import requests
from bs4 import BeautifulSoup


def normalize(u: str) -> str:
    return urldefrag(u.strip())[0]


def same_domain(seed: str, u: str) -> bool:
    a, b = urlparse(seed), urlparse(u)
    return a.netloc == b.netloc and b.scheme in ("http", "https")


def crawl(seed: str, max_depth: int, max_pages: int, timeout: float) -> set[str]:
    seen: set[str] = set()
    q = deque([(seed, 0)])
    while q and len(seen) < max_pages:
        url, depth = q.popleft()
        url = normalize(url)
        if url in seen:
            continue
        try:
            resp = requests.get(url, timeout=timeout, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            })
            if resp.status_code >= 400:
                continue
            seen.add(url)
            if depth >= max_depth:
                continue
            soup = BeautifulSoup(resp.text, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                child = urljoin(url, href)
                child = normalize(child)
                if same_domain(seed, child) and child not in seen:
                    q.append((child, depth + 1))
        except Exception:
            # Ignore fetch/parse errors to keep crawling robust
            pass
    return seen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--max-pages", type=int, default=500)
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--out", default="outputs/bfs_urls.txt")
    args = ap.parse_args()

    urls = crawl(args.start, args.depth, args.max_pages, args.timeout)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for u in sorted(urls):
            f.write(u + "\n")
    print(f"BFS discovered {len(urls)} URLs -> {out}")


if __name__ == "__main__":
    main()
