import sys
from pathlib import Path

# Ensure src/ is on sys.path for 'racen' imports
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from racen.assistant_meta import try_answer_meta_question  # noqa: E402


def test_meta_identity_returns_answer():
    ans = try_answer_meta_question("who are you?")
    assert ans is not None
    assert "RACEN" in ans
    assert "GREST" in ans


def test_meta_privacy_returns_answer():
    ans = try_answer_meta_question("how do you use my data?")
    assert ans is not None
    assert "privacy policy" in ans.lower()


def test_avoid_intercept_on_product_terms():
    # Ambiguous: contains product token; should not intercept
    ans = try_answer_meta_question("who are you iphone 13?")
    assert ans is None
