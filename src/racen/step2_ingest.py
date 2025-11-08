from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from .log import get_logger

logger = get_logger("racen.step2")


@dataclass
class Chunk:
    id: str
    text: str
    start_char: int
    end_char: int
    meta: dict


class Cleaner:
    """Lightweight Markdown cleaner to strip boilerplate and collapse whitespace."""

    BOILERPLATE_PATTERNS = [
        re.compile(
            r"^\s*<!--.*?-->\s*$",
            flags=re.MULTILINE | re.DOTALL,
        ),
        re.compile(r"^\s*\{\{.*?\}\}\s*$", flags=re.MULTILINE),
    ]

    def clean(self, md: str) -> str:
        logger.info("Cleaning markdown: start")
        text = md
        for pat in self.BOILERPLATE_PATTERNS:
            text = pat.sub("\n", text)
        # collapse 3+ newlines to 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        # strip trailing spaces
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = text.strip()
        logger.info("Cleaning markdown: done")
        return text


class Chunker:
    """Simple token-aware chunker using headings and ~tokens-per-char heuristic.

    Assumes ~4 chars per token heuristic. Uses headings (#+) as soft boundaries.
    """

    def __init__(self, max_tokens: int = 400, overlap_tokens: int = 40) -> None:
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    @staticmethod
    def _approx_tokens(s: str) -> int:
        return max(1, int(len(s) / 4))

    def chunk(self, md: str) -> List[Chunk]:
        logger.info("Chunking markdown: start")
        lines = md.splitlines()

        # identify heading indices as soft split points
        heading_idx = [
            i for i, ln in enumerate(lines) if ln.lstrip().startswith("#")
        ]
        boundaries = [0] + heading_idx + [len(lines)]

        # form sections between boundaries
        sections: List[tuple[int, int]] = []
        for a, b in zip(boundaries, boundaries[1:]):
            if a == b:
                continue
            sections.append((a, b))

        chunks: List[Chunk] = []
        cursor_chars = 0
        buf_lines: List[str] = []
        start_line = 0

        def flush(end_line: int) -> None:
            nonlocal buf_lines, cursor_chars, start_line
            if not buf_lines:
                return
            text = "\n".join(buf_lines).strip()
            start_char = (
                sum(len(line) + 1 for line in lines[:start_line])
                if start_line
                else 0
            )
            end_char = start_char + len(text)
            cid = f"c{len(chunks)+1}"
            chunks.append(
                Chunk(
                    id=cid,
                    text=text,
                    start_char=start_char,
                    end_char=end_char,
                    meta={
                        "start_line": start_line,
                        "end_line": end_line,
                    },
                )
            )
            # prepare overlap if needed
            if self.overlap_tokens > 0 and chunks:
                overlap_chars = self.overlap_tokens * 4
                join = "\n".join(buf_lines)
                tail = join[-overlap_chars:]
                buf_lines = [tail]
                cursor_chars = len(tail)
                start_line = end_line
            else:
                buf_lines = []
                cursor_chars = 0
                start_line = end_line

        for (sa, sb) in sections:
            section_text = "\n".join(lines[sa:sb])
            if not buf_lines:
                start_line = sa
            approx_tokens = self._approx_tokens(section_text)
            approx_chars = approx_tokens * 4
            if (cursor_chars + approx_chars) <= (self.max_tokens * 4):
                buf_lines.extend(lines[sa:sb])
                cursor_chars += approx_chars
            else:
                flush(sa)
                buf_lines.extend(lines[sa:sb])
                cursor_chars = approx_chars
        flush(len(lines))

        logger.info(
            f"Chunking markdown: produced {len(chunks)} chunks"
        )
        return chunks
