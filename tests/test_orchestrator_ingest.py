import types
from typing import Any, Dict, List

import pytest

from racen import orchestrator as orch


class DummyCursor:
    """Minimal cursor that records executed SQL statements for assertions.

    Args:
        executed (List[tuple[str, Any]]): Shared list to capture SQL and params.
    """

    def __init__(self, executed: List[tuple[str, Any]]) -> None:
        self._executed = executed

    def execute(self, sql: str, params: Any | None = None) -> None:
        """Record the SQL statement and parameters.

        Args:
            sql (str): SQL string.
            params (Any | None): Parameters for the SQL statement.
        """

        self._executed.append((sql.strip(), params))

    def __enter__(self) -> "DummyCursor":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False


class DummyConn:
    """Connection wrapper that returns a DummyCursor.

    Args:
        executed (List[tuple[str, Any]]): Shared list to capture SQL and params.
    """

    def __init__(self, executed: List[tuple[str, Any]]) -> None:
        self._executed = executed

    def cursor(self) -> DummyCursor:
        """Return a new DummyCursor bound to the shared executed list.

        Returns:
            DummyCursor: Cursor instance.
        """

        return DummyCursor(self._executed)

    def close(self) -> None:
        """No-op close to satisfy orchestrator expectations."""

        return None


class DummyMarkItDownClient:
    """Stub MarkItDown client that returns fixed markdown content."""

    def convert_to_markdown(self, url: str) -> str:
        """Return deterministic markdown content for tests.

        Args:
            url (str): URL passed by ingest_url (unused).

        Returns:
            str: Dummy markdown body.
        """

        return "Dummy markdown for testing."


class DummyCleaner:
    """No-op cleaner that returns the input text unchanged."""

    def clean(self, text: str) -> str:
        """Return text unchanged.

        Args:
            text (str): Input text.

        Returns:
            str: Same text.
        """

        return text


class DummyChunk:
    """Simple stand-in for a chunk with minimal metadata."""

    def __init__(self, text: str) -> None:
        """Initialize a dummy chunk.

        Args:
            text (str): Chunk text.
        """

        self.text = text
        self.start_char = 0
        self.end_char = len(text)
        self.meta: Dict[str, int] = {"start_line": 1, "end_line": 1}


class DummyChunker:
    """Chunker stub that returns a single DummyChunk covering the full text."""

    def __init__(self, max_tokens: int = 400, overlap_tokens: int = 40) -> None:
        """Initialize the dummy chunker (arguments unused).

        Args:
            max_tokens (int): Unused.
            overlap_tokens (int): Unused.
        """

        self._max_tokens = max_tokens
        self._overlap_tokens = overlap_tokens

    def chunk(self, text: str) -> List[DummyChunk]:
        """Return a single DummyChunk for the given text.

        Args:
            text (str): Cleaned text.

        Returns:
            List[DummyChunk]: List with one chunk.
        """

        return [DummyChunk(text)]


class DummyEmbedder:
    """Embedder stub that returns a tiny fixed vector and model name."""

    def embed(self, id: str, text: str, metadata: Dict[str, str]) -> Any:  # noqa: ARG002
        """Return a simple namespace with vector and model.

        Args:
            id (str): Chunk identifier.
            text (str): Chunk text.
            metadata (Dict[str, str]): Metadata (unused).

        Returns:
            Any: Object with vector and model attributes.
        """

        return types.SimpleNamespace(vector=[0.1, 0.2], model="dummy-model")


def test_ingest_url_deletes_existing_document_before_upsert(monkeypatch: pytest.MonkeyPatch) -> None:
    """ingest_url should delete any existing document row before upserting.

    This ensures that re-ingesting the same URL behaves as a full overwrite and
    does not keep historical chunks/embeddings for that document id.
    """

    executed_sql: List[tuple[str, Any]] = []
    conn = DummyConn(executed_sql)

    # Patch orchestrator dependencies so we do not hit the real DB or network.
    monkeypatch.setattr(orch, "get_conn", lambda cfg=None: conn)
    monkeypatch.setattr(orch, "ensure_schema", lambda c, embedding_dim=256: None)
    monkeypatch.setattr(orch, "MarkItDownClient", DummyMarkItDownClient)
    monkeypatch.setattr(orch, "Cleaner", DummyCleaner)
    monkeypatch.setattr(orch, "Chunker", DummyChunker)
    monkeypatch.setattr(orch, "OpenAIEmbedder", DummyEmbedder)

    recorded_doc_ids: List[str] = []

    def fake_upsert_document(connection: DummyConn, *, doc_id: str, source: str) -> None:  # noqa: ARG001
        """Capture the doc_id that ingest_url passes to upsert_document.

        Args:
            connection (DummyConn): Dummy connection.
            doc_id (str): Document identifier.
            source (str): Source URL.
        """

        recorded_doc_ids.append(doc_id)

    # Remaining DB helpers are no-ops for this behavioral test.
    monkeypatch.setattr(orch, "upsert_document", fake_upsert_document)
    monkeypatch.setattr(orch, "upsert_chunk", lambda *args, **kwargs: None)
    monkeypatch.setattr(orch, "upsert_embedding", lambda *args, **kwargs: None)
    monkeypatch.setattr(orch, "embedding_exists", lambda connection, chunk_id: False)  # noqa: ARG001

    url = "https://grest.in/products/refurbished-apple-iphone-13?variant=123456"

    result = orch.ingest_url(url, embedding_dim=8)

    # We should have exactly one upsert_document call with the returned doc_id.
    assert recorded_doc_ids == [result.doc_id]

    # And we should have issued a DELETE FROM documents for that doc_id before insert.
    delete_statements = [sql for (sql, params) in executed_sql if sql.startswith("DELETE FROM documents")]
    assert delete_statements, "ingest_url must delete existing document rows before upsert"
