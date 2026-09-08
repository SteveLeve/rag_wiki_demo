#!/usr/bin/env python3
"""Verify the model catalog against a real Ollama server.

The catalog in ragkit.models asserts a dimension for every embedding model. A
wrong number there is dangerous rather than merely untidy: the registry drives
table DDL, so a bad dimension produces a table that silently rejects or truncates
vectors. This script pulls each model and measures its real output.

    python scripts/preflight.py            # check only
    python scripts/preflight.py --pull     # pull anything missing first
"""
from __future__ import annotations

import argparse
import subprocess
import sys

from ragkit.models import EMBEDDING_MODELS, LANGUAGE_MODELS


def installed_tags() -> set[str]:
    out = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True).stdout
    tags = set()
    for line in out.splitlines()[1:]:
        if line.strip():
            tag = line.split()[0]
            tags.add(tag)
            if tag.endswith(":latest"):
                tags.add(tag[: -len(":latest")])
    return tags


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true", help="pull missing models before checking")
    args = ap.parse_args()

    try:
        import ollama
    except ImportError:
        print("ollama package not installed: pip install -e '.[test]'", file=sys.stderr)
        return 2

    if not hasattr(ollama, "embed"):
        print("ollama package is too old: ragkit needs >=0.5 for embed() and batch input.", file=sys.stderr)
        return 2

    have = installed_tags()
    failures = 0

    print("Embedding models")
    for alias, model in EMBEDDING_MODELS.items():
        if model.ollama_tag not in have:
            if not args.pull:
                print(f"  ? {alias:20s} {model.ollama_tag:22s} NOT PULLED (re-run with --pull)")
                failures += 1
                continue
            print(f"  … pulling {model.ollama_tag}")
            subprocess.run(["ollama", "pull", model.ollama_tag], check=True)

        vec = ollama.embed(model=model.ollama_tag, input="dimension probe")["embeddings"][0]
        ok = len(vec) == model.dimension
        mark = "✓" if ok else "✗"
        detail = f"dim={len(vec)}" + ("" if ok else f"  CATALOG SAYS {model.dimension}")
        print(f"  {mark} {alias:20s} {model.ollama_tag:22s} {detail}")
        failures += 0 if ok else 1

    print("\nLanguage models")
    for tag in LANGUAGE_MODELS:
        present = tag in have
        if not present and args.pull:
            print(f"  … pulling {tag}")
            subprocess.run(["ollama", "pull", tag], check=True)
            present = True
        print(f"  {'✓' if present else '?'} {tag:22s} {'available' if present else 'NOT PULLED'}")
        failures += 0 if present else 1

    print(f"\n{failures} problem(s)" if failures else "\nCatalog matches reality.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
