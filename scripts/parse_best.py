import ast
import glob
import re
import sys
from pathlib import Path


PATTERN = re.compile(r"Best Test Results across All Epochs:\s*(\{.*\})")


def parse_log(path: Path):
    best = None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = PATTERN.search(line)
        if match:
            best = ast.literal_eval(match.group(1))
    return best


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/parse_best.py experiments/logs/*.log")
    paths = []
    for item in sys.argv[1:]:
        matches = glob.glob(item)
        paths.extend(matches if matches else [item])
    for item in paths:
        path = Path(item)
        metrics = parse_log(path)
        print(f"{path}: {metrics}")


if __name__ == "__main__":
    main()
