from __future__ import annotations

import collections
import time
import urllib.parse
import urllib.robotparser as robotparser
from dataclasses import dataclass
from typing import List, Set, Sequence

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .log import get_logger

logger = get_logger("racen.crawler")


@dataclass
class CrawlConfig:
    start_url: str
    max_pages: int = 500
    same_domain: bool = True
    rate_limit_rps: float = 1.0  # requests per second
    timeout: int = 20
    user_agent: str = "RACEN-Crawler/0.1 (+https://grest.in)"
    include_patterns: Sequence[str] | None = None
    exclude_patterns: Sequence[str] | None = None
    retries: int = 3


def normalize_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    # drop fragment, normalize scheme/host
    norm = parsed._replace(fragment="")
    # ensure lowercase scheme/hostname
    scheme = (norm.scheme or "http").lower()
    netloc = norm.netloc.lower()
    path = norm.path or "/"
    query = norm.query
    return urllib.parse.urlunsplit((scheme, netloc, path, query, ""))


def same_domain(url: str, origin: str) -> bool:
    a = urllib.parse.urlsplit(url).netloc.lower()
    b = urllib.parse.urlsplit(origin).netloc.lower()
    return a == b or a.endswith("." + b) or b.endswith("." + a)


def allowed_by_robots(target_url: str, ua: str) -> bool:
    parsed = urllib.parse.urlsplit(target_url)
    robots_url = urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, "/robots.txt", "", "")
    )
    rp = robotparser.RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(ua, target_url)
    except Exception:
        # be conservative: allow if robots can't be fetched
        return True


def extract_links(html: str, base_url: str) -> List[str]:
    links: List[str] = []
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        href = a.get("href").strip()
        # resolve relative URLs
        abs_url = urllib.parse.urljoin(base_url, href)
        links.append(abs_url)
    return links


def _build_session(timeout: int, ua: str, retries: int) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=max(0, retries),
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update({"User-Agent": ua})
    s.timeout = timeout  # type: ignore[attr-defined]
    return s


def _should_visit(url: str, origin: str, cfg: CrawlConfig) -> bool:
    # Always allow the origin/start URL
    try:
        if normalize_url(url) == normalize_url(origin):
            return True
    except Exception:
        pass
    if cfg.same_domain and not same_domain(url, origin):
        return False
    path = urllib.parse.urlsplit(url).path.lower()
    # default excludes
    default_ex = ["cart", "checkout", "admin", "login", "signup"]
    if any(tok in path for tok in default_ex):
        return False
    if cfg.exclude_patterns:
        for pat in cfg.exclude_patterns:
            try:
                if pat and pat.lower() in url.lower():
                    return False
            except Exception:
                continue
    if cfg.include_patterns:
        # at least one include must match
        ok = any(
            pat.lower() in url.lower()
            for pat in cfg.include_patterns
            if pat
        )
        if not ok:
            return False
    return True


def crawl(cfg: CrawlConfig) -> List[str]:
    start = normalize_url(cfg.start_url)
    domain_root = start

    # Try to resolve start; fallback to www. and http if DNS/Scheme issues
    candidates = [start]
    parsed = urllib.parse.urlsplit(start)
    if not parsed.netloc.startswith("www."):
        candidates.append(
            urllib.parse.urlunsplit(
                (
                    parsed.scheme,
                    "www." + parsed.netloc,
                    parsed.path,
                    parsed.query,
                    "",
                )
            )
        )
    if parsed.scheme == "https":
        candidates.append(
            urllib.parse.urlunsplit(
                ("http", parsed.netloc, parsed.path, parsed.query, "")
            )
        )
    start_resolved = None
    session = _build_session(cfg.timeout, cfg.user_agent, cfg.retries)
    for cand in candidates:
        try:
            logger.info(f"Crawl fetch (probe): {cand}")
            r = session.get(cand, timeout=cfg.timeout)
            if r.status_code < 500:
                start_resolved = cand
                break
        except Exception as e:
            logger.info(f"Probe failed for {cand}: {e}")
            continue
    if start_resolved:
        start = normalize_url(start_resolved)
        domain_root = start
    else:
        # No probe succeeded. We'll continue optimistically using the original
        # candidate and attempt sitemap-based seeding across candidates.
        logger.warning(
            (
                f"Unable to resolve start URL after probes: {start} — "
                "attempting sitemap-based seeding"
            )
        )

    if cfg.same_domain and not allowed_by_robots(start, cfg.user_agent):
        logger.warning(f"Blocked by robots.txt: {start}")
        return []

    q: collections.deque[str] = collections.deque([start])
    seen: Set[str] = {start}
    out: List[str] = []

    delay = 1.0 / cfg.rate_limit_rps if cfg.rate_limit_rps > 0 else 0.0

    # Seed from sitemap if present (try across all candidates if needed)
    seeded = 0
    for cand in candidates:
        try:
            site = urllib.parse.urlsplit(cand)
            sm_url = urllib.parse.urlunsplit(
                (site.scheme, site.netloc, "/sitemap.xml", "", "")
            )
            logger.info(f"Sitemap probe: {sm_url}")
            sm_resp = session.get(sm_url, timeout=cfg.timeout)
            if sm_resp.status_code == 200 and "xml" in (
                sm_resp.headers.get("Content-Type", "").lower()
            ):
                # naive parse: collect <loc>...</loc>
                locs = []
                for m in sm_resp.text.split("<loc>")[1:]:
                    loc = m.split("</loc>", 1)[0].strip()
                    if loc:
                        locs.append(loc)
                for loc in locs:
                    u = normalize_url(loc)
                    if u not in seen and _should_visit(u, domain_root, cfg):
                        seen.add(u)
                        q.append(u)
                        seeded += 1
                logger.info(
                    f"Sitemap seed added: {len(locs)} urls from {sm_url}"
                )
                # If we didn't have a resolved start, set domain_root based on this cand
                if not start_resolved:
                    start = normalize_url(cand)
                    domain_root = start
        except Exception as e:
            logger.info(f"Sitemap skip for {cand}: {e}")

    # If we couldn't seed anything and we have a start URL, enqueue it anyway
    if not q:
        q.append(start)
        seen.add(start)

    while q and len(out) < cfg.max_pages:
        url = q.popleft()
        try:
            if not _should_visit(url, domain_root, cfg):
                continue
            if not allowed_by_robots(url, cfg.user_agent):
                logger.info(f"robots disallow: {url}")
                continue
            logger.info(f"Crawl fetch: {url}")
            resp = session.get(url, timeout=cfg.timeout)
            ctype = resp.headers.get("Content-Type", "").lower()
            if "text/html" not in ctype:
                logger.info(f"Skip non-HTML: {url} ({ctype})")
                time.sleep(delay)
                continue
            html = resp.text
            out.append(url)
            # enqueue new links
            for link in extract_links(html, url):
                n = normalize_url(link)
                if n not in seen and _should_visit(n, domain_root, cfg):
                    seen.add(n)
                    q.append(n)
        except Exception as e:
            logger.warning(f"Crawl error for {url}: {e}")
        finally:
            if delay > 0:
                time.sleep(delay)

    logger.info(f"Crawl complete: {len(out)} pages")
    return out
