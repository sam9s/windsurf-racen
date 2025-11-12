import os
import requests
from typing import Optional
from ..utils.env import load_env


def convert_to_markdown(
    uri: str,
    base_url: Optional[str] = None,
    timeout: float = 30.0,
) -> str:
    """
    Call the local MarkItDown HTTP service to convert a URI to markdown.

    Args:
        uri (str): http/https/file/data URI to convert.
        base_url (Optional[str]): Base URL of the service; falls back to
            MARKITDOWN_HTTP_URL.
        timeout (float): Request timeout in seconds.

    Returns:
        str: Markdown content.
    """
    load_env()
    service = base_url or os.getenv(
        "MARKITDOWN_HTTP_URL", "http://127.0.0.1:8082"
    )
    resp = requests.post(
        f"{service}/convert", json={"uri": uri}, timeout=timeout
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("markdown", "")
