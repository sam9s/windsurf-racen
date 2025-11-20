from __future__ import annotations

from typing import List

from scripts import build_blog_yaml as bb


def test_extract_blog_urls_from_html_filters_and_normalizes() -> None:
    """extract_blog_urls_from_html should keep only distinct article URLs.

    The helper should:
    - resolve relative URLs against the index URL,
    - keep only grest.in /blogs/news/... links,
    - drop the index page itself,
    - deduplicate and sort results.
    """

    index_url = "https://grest.in/blogs/news"
    html = """
    <html>
      <body>
        <a href="/blogs/news/how-we-refurbish-iphones">Article 1</a>
        <a href="https://grest.in/blogs/news/why-buy-refurbished">Article 2</a>
        <a href="/blogs/news/how-we-refurbish-iphones">Duplicate Article 1</a>
        <a href="/blogs/news">News index</a>
        <a href="/products/refurbished-apple-iphone-13">Product page</a>
        <a href="https://example.com/blogs/news/external">External blog</a>
      </body>
    </html>
    """

    urls: List[str] = bb.extract_blog_urls_from_html(html, index_url=index_url)

    assert urls == [
        "https://grest.in/blogs/news/how-we-refurbish-iphones",
        "https://grest.in/blogs/news/why-buy-refurbished",
    ]
