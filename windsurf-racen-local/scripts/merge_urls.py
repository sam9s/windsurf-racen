import argparse
from pathlib import Path
from urllib.parse import urldefrag


def normalize(u: str) -> str:
    return urldefrag(u.strip())[0]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+", help="Input url list files (txt)")
    p.add_argument("-o", "--out", required=True, help="Output txt path")
    args = p.parse_args()

    seen: set[str] = set()
    for path in args.inputs:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                u = normalize(line)
                if u:
                    seen.add(u)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for u in sorted(seen):
            f.write(u + "\n")

    print(f"Merged {len(args.inputs)} files -> {len(seen)} unique URLs at {out}")


if __name__ == "__main__":
    main()
