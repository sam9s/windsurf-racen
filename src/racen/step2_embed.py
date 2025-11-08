from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import List, Optional

import requests

from .log import get_logger

logger = get_logger("racen.embed")


@dataclass
class EmbeddingResult:
    id: str
    vector: List[float]
    model: str
    dim: int
    meta: dict


class OpenAIEmbedder:
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        timeout: int = 30,
    ) -> None:
        self.model = model
        self.timeout = timeout
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv(
            "OPENAI_BASE_URL",
            "https://api.openai.com/v1",
        )

    def embed(
        self,
        *,
        id: str,
        text: str,
        metadata: Optional[dict] = None,
    ) -> EmbeddingResult:
        if not self.api_key:
            logger.warning(
                "OPENAI_API_KEY not set; using deterministic local fallback embedding"
            )
            vec = self._fallback_embed(text)
            return EmbeddingResult(
                id=id,
                vector=vec,
                model=f"{self.model}-fallback",
                dim=len(vec),
                meta=metadata or {},
            )

        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "input": text,
        }
        logger.info(
            f"Requesting embeddings (model={self.model}, len={len(text)} chars)"
        )
        resp = requests.post(
            url,
            headers=headers,
            data=json.dumps(payload),
            timeout=self.timeout,
        )
        if resp.status_code != 200:
            logger.error(
                f"OpenAI embeddings error {resp.status_code}: {resp.text[:200]}"
            )
            raise RuntimeError(
                f"OpenAI embeddings failed: {resp.status_code}"
            )
        data = resp.json()
        vec = data["data"][0]["embedding"]
        return EmbeddingResult(
            id=id,
            vector=vec,
            model=self.model,
            dim=len(vec),
            meta=metadata or {},
        )

    @staticmethod
    def _fallback_embed(text: str, dim: int = 256) -> List[float]:
        # Produce a pseudo-embedding by hashing; stable and deterministic for smoke tests
        h = hashlib.sha256(text.encode("utf-8")).digest()
        # Repeat hash to fill dim
        buf = (h * ((dim // len(h)) + 1))[:dim]
        # Normalize bytes 0-255 to 0..1 floats
        return [b / 255.0 for b in buf]
