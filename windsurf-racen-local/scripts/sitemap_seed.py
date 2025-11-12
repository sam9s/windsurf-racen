import argparse
from urllib.parse import urljoin
from pathlib import Path
import requests
from lxml import etree


def fetch_xml(url: str) -> bytes:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.content


def parse_sitemap_xml(xml_bytes: bytes) -> list[str]:
    urls: list[str] = []
    root = etree.fromstring(xml_bytes)
    ns = {
        "sm": "http://www.sitemaps.org/schemas/sitemap/0.9",
        "xhtml": "http://www.w3.org/1999/xhtml",
    }
    # <urlset><url><loc>...
    for loc in root.xpath("//sm:url/sm:loc/text()", namespaces=ns):
        urls.append(loc.strip())
    # Or <sitemapindex><sitemap><loc> (we return empty here; caller can handle chaining)
    return urls


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Root like https://grest.in/")
    parser.add_argument("--path", default="sitemap.xml", help="Sitemap path")
    parser.add_argument("--out", default="outputs/sitemap_urls.txt", help="Output file")
    args = parser.parse_args()

    root = args.root.rstrip("/") + "/"
    sitemap_url = urljoin(root, args.path)

    try:
        xml_bytes = fetch_xml(sitemap_url)
        urls = parse_sitemap_xml(xml_bytes)
    except Exception as e:
        print(f"[WARN] Failed primary sitemap {sitemap_url}: {e}")
        urls = []

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for u in urls:
            f.write(u + "\n")
    print(f"Wrote {len(urls)} URLs to {out}")


if __name__ == "__main__":
    main()
