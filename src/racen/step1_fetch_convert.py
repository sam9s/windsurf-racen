from __future__ import annotations

import pathlib
from typing import Optional
import os

import requests
from markdownify import markdownify as md

from .log import get_logger

logger = get_logger("racen.step1")


class MarkItDownClient:
    """
    Placeholder interface for MarkItDown MCP `convert_to_markdown`.
    Fallback uses local HTML->Markdown conversion.
    """

    def __init__(self) -> None:
        # Optional HTTP endpoint for MarkItDown MCP-like service
        self.mcp_url = os.getenv("MARKITDOWN_HTTP_URL")

    def convert_to_markdown(
        self,
        *,
        url: Optional[str] = None,
        file_path: Optional[str] = None,
        html: Optional[str] = None,
    ) -> str:
        if url:
            # Prefer remote MarkItDown MCP if configured
            if self.mcp_url:
                try:
                    resp = requests.post(
                        self.mcp_url,
                        json={"url": url},
                        timeout=30,
                        headers={"Content-Type": "application/json"},
                    )
                    if resp.status_code == 200:
                        return resp.text.strip()
                    logger.info(
                        f"MarkItDown MCP HTTP returned {resp.status_code}; falling back"
                    )
                except Exception as e:
                    logger.info(f"MarkItDown MCP HTTP failed: {e}; falling back")
            html_text = fetch_url(url)
            return convert_html_to_markdown(html_text, base_url=url)
        if file_path:
            # Remote MCP not attempted for local files in this simple hook
            return file_to_markdown(file_path)
        if html is not None:
            # If MCP URL present, allow posting raw HTML as body
            if self.mcp_url:
                try:
                    resp = requests.post(
                        self.mcp_url,
                        data=html.encode("utf-8"),
                        timeout=30,
                        headers={"Content-Type": "text/html; charset=utf-8"},
                    )
                    if resp.status_code == 200:
                        return resp.text.strip()
                    logger.info(
                        f"MarkItDown MCP HTTP returned {resp.status_code}; falling back"
                    )
                except Exception as e:
                    logger.info(f"MarkItDown MCP HTTP failed: {e}; falling back")
            return convert_html_to_markdown(html)
        raise ValueError("One of url, file_path, or html must be provided")


def fetch_url(url: str, timeout: int = 20) -> str:
    logger.info(f"Fetching URL: {url}")
    resp = requests.get(url, timeout=timeout, headers={
        "User-Agent": "RACEN/0.1 (+https://grest.example)"
    })
    resp.raise_for_status()
    logger.info(f"Fetched {len(resp.text)} bytes")
    return resp.text


def convert_html_to_markdown(html: str, base_url: Optional[str] = None) -> str:
    """Convert HTML to Markdown using markdownify.
    """
    logger.info("Converting HTML -> Markdown")
    md_text = md(
        html,
        heading_style="ATX",
        strip=['script', 'style', 'noscript'],
        autolinks=True,
    )
    return md_text.strip()


def file_to_markdown(path: str) -> str:
    p = pathlib.Path(path)
    logger.info(f"Converting file -> Markdown: {p}")
    if not p.exists():
        raise FileNotFoundError(str(p))

    ext = p.suffix.lower()
    text = p.read_text(encoding="utf-8", errors="ignore")

    if ext in {".htm", ".html"}:
        return convert_html_to_markdown(text)
    if ext in {".md", ".markdown"}:
        return text

    # Fallback: treat as plain text and wrap in code fence
    logger.warning(f"Unknown extension {ext}, treating as plain text")
    return text
