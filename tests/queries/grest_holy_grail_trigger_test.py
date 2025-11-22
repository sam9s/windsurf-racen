from __future__ import annotations

import sys

from grest_holy_grail_test_file import run


if __name__ == "__main__":
    code = run(quiet=True)
    print(f"Exit code: {code}")
    sys.exit(code)
