import sys
from pathlib import Path
import argparse

# Ensure src is importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcp_clients.markitdown_client import convert_to_markdown  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("uri", help="http/https/file/data URI to convert")
    args = parser.parse_args()

    md = convert_to_markdown(args.uri)
    print("--- Markdown (first 400 chars) ---")
    print(md[:400])


if __name__ == "__main__":
    main()
