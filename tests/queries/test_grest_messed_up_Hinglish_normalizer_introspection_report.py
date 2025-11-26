from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
QUERIES_DIR = ROOT / "tests" / "queries"
if str(QUERIES_DIR) not in sys.path:
    sys.path.insert(0, str(QUERIES_DIR))

import grest_messed_up_Hinglish_normalizer_introspection_report as rep  # type: ignore[import]


def test_load_queries_parses_plain_and_top_k(tmp_path) -> None:
    """_load_queries should handle plain lines and `query||k` format.

    It must ignore comments/blank lines and default top_k to 10 when missing
    or invalid.
    """

    content = "\n".join(
        [
            "sabse sasta iphone||5",
            "   basic hinglish query   ",
            "# comment line",
            "",
            "last one||notint",
        ]
    )
    path: Path = tmp_path / "hinglish_queries.txt"
    path.write_text(content, encoding="utf-8")

    out = rep._load_queries(path)

    assert out[0] == ("sabse sasta iphone", 5)
    assert out[1] == ("basic hinglish query", 10)
    assert out[2] == ("last one", 10)
    assert len(out) == 3
