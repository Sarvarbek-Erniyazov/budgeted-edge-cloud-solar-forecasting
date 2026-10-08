"""Fail if any staged text file contains an absolute local path (user name leak).

Used as a git pre-commit hook (scripts/install_hooks.sh), so a commit with a leaked path is refused no
matter how the commit command was chained. Exit code 1 lists the offending files and lines.
"""
from __future__ import annotations

import re
import subprocess
import sys

PATTERNS = [re.compile(r"[A-Za-z]:\\+Users\\+[^\\/\s]+", re.I), re.compile(r"/[a-z]/Users/[^/\s]+", re.I),
            re.compile(r"/home/[^/\s]+/"), re.compile(r"/Users/[^/\s]+/")]


def staged_files() -> list[str]:
    out = subprocess.run(["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
                         capture_output=True, text=True, check=True).stdout
    return [f for f in out.splitlines() if f]


def main(files: list[str]) -> int:
    bad = []
    for f in files:
        blob = subprocess.run(["git", "show", f":{f}"], capture_output=True).stdout
        if b"\0" in blob[:8000]:
            continue                                   # binary
        for n, line in enumerate(blob.decode("utf-8", "replace").splitlines(), 1):
            if any(p.search(line) for p in PATTERNS) and "check_paths.py" not in f:
                bad.append(f"{f}:{n}: {line.strip()[:120]}")
    if bad:
        print("Refusing commit: absolute local paths found in staged files:\n" + "\n".join(bad), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or staged_files()))
