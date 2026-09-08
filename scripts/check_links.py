#!/usr/bin/env python3
"""Verify every relative markdown link resolves to a file that exists.

Run from the repo root:  python scripts/check_links.py
Exits non-zero listing each broken link, so CI can gate on it.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LINK = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")
SKIP_PREFIXES = ("http://", "https://", "mailto:", "#")


def broken_links_in(md: Path) -> list[tuple[int, str, str]]:
    """Return (line_no, link_text, target) for each unresolvable link."""
    found = []
    for lineno, line in enumerate(md.read_text(encoding="utf-8").splitlines(), 1):
        for text, target in LINK.findall(line):
            target = target.split()[0].strip()  # drop optional "title"
            if target.startswith(SKIP_PREFIXES) or not target:
                continue
            path_part = target.split("#", 1)[0]
            if not path_part:  # pure in-page anchor
                continue
            if not (md.parent / path_part).resolve().exists():
                found.append((lineno, text, target))
    return found


def main() -> int:
    total = 0
    for md in sorted(ROOT.rglob("*.md")):
        if any(part in {".venv", ".git", "node_modules"} for part in md.parts):
            continue
        for lineno, text, target in broken_links_in(md):
            rel = md.relative_to(ROOT)
            print(f"{rel}:{lineno}: broken link [{text}]({target})")
            total += 1
    print(f"\n{total} broken link(s)" if total else "\nAll markdown links resolve.")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
