#!/usr/bin/env python3
"""Structural notebook edits, as idempotent passes.

Each pass can be run alone and reviewed with `git diff`. Running a pass twice is
a no-op. Deliberately not jupytext: a permanent text pairing doubles what a
learner browsing this repo has to look at.

    python scripts/nb_apply.py --list
    python scripts/nb_apply.py strip_outputs
    python scripts/nb_apply.py all --dry-run
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TIERS = ("foundation", "intermediate", "advanced-techniques", "evaluation-lab")

# Human titles for notebooks that never had an H1.
TITLES = {
    "00-load-or-generate-pattern": "Foundation 00: The Load-or-Generate Pattern",
    "03-loading-and-reusing-embeddings": "Intermediate 03: Loading and Reusing Embeddings",
    "05-reranking": "Advanced 05: Reranking with a Cross-Encoder",
    "06-query-expansion": "Advanced 06: Query Expansion",
    "07-hybrid-search": "Advanced 07: Hybrid Search (Vector + BM25)",
    "08-semantic-chunking-and-metadata": "Advanced 08: Semantic Chunking and Metadata Filtering",
    "09-citation-tracking": "Advanced 09: Citation Tracking",
    "10-combined-advanced-rag": "Advanced 10: Combining Every Technique",
    "01-create-ground-truth-human-in-loop": "Evaluation 01: Building Ground Truth (Human in the Loop)",
    "02-evaluation-metrics-framework": "Evaluation 02: A Metrics Framework",
    "03-baseline-and-comparison": "Evaluation 03: Baselines and Comparison",
    "04-experiment-dashboard": "Evaluation 04: The Experiment Dashboard",
    "05-supplemental-embedding-analysis": "Evaluation 05: Digging Into Retrieval Quality",
}

# Old model identifier -> new. Ordered: longest/most specific first.
MODEL_SWAPS = [
    ("hf.co/CompendiumLabs/bge-base-en-v1.5-gguf", "nomic-embed-text"),
    ("hf.co/CompendiumLabs/bge-small-en-v1.5-gguf", "all-minilm"),
    ("hf.co/CompendiumLabs/bge-large-en-v1.5-gguf", "mxbai-embed-large"),
    ("hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF", "llama3.2:3b"),
    ("bge_base_en_v1.5", "nomic_embed_text"),
    ("bge_base_en_v1_5", "nomic_embed_text"),
    ("bge_small_en_v1.5", "all_minilm_l6_v2"),
    ("bge_small_en_v1_5", "all_minilm_l6_v2"),
    ("all-minilm-l6-v2", "all_minilm_l6_v2"),
    ("llama3.2:1b", "llama3.2:3b"),
]

IMPORTS_MARKER = "# --- ragkit: shared utilities ---"
IMPORTS_CELL = f'''{IMPORTS_MARKER}
# These live in src/ragkit/ so every notebook uses one implementation. Open the
# module if you want to read it -- it is meant to be read. Anything a notebook is
# *teaching* stays written out inline below; see AGENTS.md for the rule.
from ragkit import config, db, registry
from ragkit.db import table_name_for
from ragkit.embed import embed_one, embed_texts
from ragkit.store import VectorStore

print(config.describe())
'''


def cell_source(cell: dict) -> str:
    src = cell.get("source", [])
    if isinstance(src, str):
        return src
    if src and not any(line.endswith("\n") for line in src[:-1]):
        return "\n".join(src)
    return "".join(src)


def set_source(cell: dict, text: str) -> None:
    lines = text.splitlines(keepends=True)
    cell["source"] = lines


def notebooks(tiers=TIERS) -> list[Path]:
    return sorted(p for t in tiers for p in (ROOT / t).glob("*.ipynb"))


# ---------------------------------------------------------------- passes


def strip_outputs(nb: dict, path: Path) -> bool:
    """Remove committed outputs and repair required code-cell fields.

    Some cells in this repo were missing `execution_count` entirely, which
    nbformat rejects as invalid -- so those notebooks could never be validated,
    let alone executed by papermill.
    """
    changed = False
    # Cell ids only exist in nbformat 4.5+. Adding one to an older notebook makes
    # it invalid, so bring the notebook forward first.
    if nb.get("nbformat") == 4 and nb.get("nbformat_minor", 0) < 5:
        nb["nbformat_minor"] = 5
        changed = True

    seen_ids: set[str] = set()
    for i, cell in enumerate(nb.get("cells", [])):
        # Duplicate cell ids are a hard error in newer nbformat.
        cid = cell.get("id")
        if not cid or cid in seen_ids:
            cell["id"] = f"cell-{i:03d}"
            changed = True
        seen_ids.add(cell["id"])

        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs"):
            cell["outputs"] = []
            changed = True
        cell.setdefault("outputs", [])
        if cell.get("execution_count", "missing") != None:  # noqa: E711 - distinguishes absent
            cell["execution_count"] = None
            changed = True
    return changed


def normalize_source(nb: dict, path: Path) -> bool:
    """Repair cells whose source lines were stored without trailing newlines.

    foundation/01 had one such cell; concatenating it produced `import osprint(`,
    which breaks every AST-based tool that touches this repo.
    """
    changed = False
    for cell in nb.get("cells", []):
        src = cell.get("source", [])
        if isinstance(src, list) and len(src) > 1 and not any(l.endswith("\n") for l in src[:-1]):
            set_source(cell, "\n".join(src))
            changed = True
    return changed


def add_title(nb: dict, path: Path) -> bool:
    """Ensure the first cell is a markdown H1."""
    cells = nb.get("cells", [])
    if cells:
        first = cells[0]
        if first.get("cell_type") == "markdown" and cell_source(first).lstrip().startswith("# "):
            return False
    title = TITLES.get(path.stem)
    if not title:
        return False
    cells.insert(0, {
        "cell_type": "markdown",
        "metadata": {},
        "source": [f"# {title}\n"],
    })
    return True


def swap_models(nb: dict, path: Path) -> bool:
    """Replace retired model identifiers with catalog names."""
    changed = False
    for cell in nb.get("cells", []):
        src = cell_source(cell)
        new = src
        for old, replacement in MODEL_SWAPS:
            new = new.replace(old, replacement)
        if new != src:
            set_source(cell, new)
            changed = True
    return changed


def add_imports(nb: dict, path: Path) -> bool:
    """Insert the standard ragkit import cell after the title."""
    cells = nb.get("cells", [])
    if any(IMPORTS_MARKER in cell_source(c) for c in cells):
        return False
    if path.stem.startswith("INDEX"):
        return False
    # foundation/01 teaches RAG with no dependencies at all; leave it alone.
    if path.stem == "01-basic-rag-in-memory":
        return False
    insert_at = 1 if cells and cells[0].get("cell_type") == "markdown" else 0
    cells.insert(insert_at, {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": IMPORTS_CELL.splitlines(keepends=True),
    })
    return True



# Table names built by hand, in six different notebooks, with two different
# normalizations. This is the bug that split one model across two registry rows.
TABLE_NAME_PATTERNS = [
    (re.compile(r"""f['"]embeddings_\{(\w+)\.replace\(["']\.["'],\s*["']_["']\)"""
                r"""(?:\.replace\(["']-["'],\s*["']_["']\))?\}['"]"""),
     r"table_name_for(\1)"),
]


def canonical_tables(nb: dict, path: Path) -> bool:
    """Replace hand-rolled table-name derivation with db.table_name_for()."""
    changed = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = new = cell_source(cell)
        for pattern, replacement in TABLE_NAME_PATTERNS:
            new = pattern.sub(replacement, new)
        if new != src:
            set_source(cell, new)
            changed = True
    return changed


PRESERVE_PROMPT = """            while True:
                response = input('\\nDo you want to (p)reserve existing data or (o)verwrite it? [p/o]: ').lower().strip()
                if response in ['p', 'preserve']:
                    preserve_existing = True
                    break
                elif response in ['o', 'overwrite']:
                    preserve_existing = False
                    break
                else:
                    print('Please enter "p" for preserve or "o" for overwrite')"""

PRESERVE_DEFAULT = """            # Default to preserving. This used to prompt on stdin, which hangs
            # papermill forever -- pass preserve_existing=False to overwrite.
            preserve_existing = True
            print('Preserving by default. Pass preserve_existing=False to rebuild.')"""


def remove_input_prompts(nb: dict, path: Path) -> bool:
    """Replace interactive input() prompts with an explicit default.

    input() blocks forever under papermill, so any notebook containing one could
    never be executed in CI -- which is a large part of why notebook tests
    'failed'.
    """
    changed = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell_source(cell)
        if "input(" not in src:
            continue
        new = src.replace(PRESERVE_PROMPT, PRESERVE_DEFAULT)
        if new == src:
            # No generic fallback here on purpose. An earlier version substituted
            # `response = ''`, which matches no branch of the surrounding
            # `while True:` and spins forever -- strictly worse than the input()
            # it replaced. Prompts differ enough that each needs a deliberate
            # default, so refuse rather than guess.
            raise SystemExit(
                f"{path}: unrecognised input() prompt. Add its safe default to "
                "PRESERVE_PROMPT/PRESERVE_DEFAULT rather than letting the codemod guess; "
                "a wrong default here produces an infinite loop under papermill."
            )
        if new != src:
            set_source(cell, new)
            changed = True
    return changed


def dynamic_dimension(nb: dict, path: Path) -> bool:
    """Replace hardcoded vector(768) with the registry-supplied dimension."""
    changed = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell_source(cell)
        if not re.search(r"vector\(\s*768\s*\)", src):
            continue
        new = re.sub(r"vector\(\s*768\s*\)", "vector({self.dimension})", src)
        # The class must know its dimension; take it from the registry at init.
        new = new.replace(
            "        self.table_name = table_name\n",
            "        self.table_name = table_name\n"
            "        # Read the width from the registry instead of assuming 768, so the\n"
            "        # same class works for 384-, 768- and 1024-dimensional models.\n"
            "        self.dimension = dimension\n",
        )
        new = new.replace(
            "    def __init__(self, config, table_name, preserve_existing=None):",
            "    def __init__(self, config, table_name, dimension, preserve_existing=None):",
        )
        if new != src:
            set_source(cell, new)
            changed = True
    return changed



# Thirteen notebooks hardcoded a connection dict pointing at port 5432. That port
# is routinely occupied by an unrelated project's PostgreSQL, and these notebooks
# create and drop tables -- so the default must come from configuration.
# Capture the leading indentation: these dicts are sometimes nested inside an
# `if` block, and emitting an unindented replacement produces an IndentationError.
PG_CONFIG_RE = re.compile(
    r"^([ \t]*)POSTGRES_CONFIG\s*=\s*\{[^}]*?'host'[^}]*?\}", re.S | re.M
)


def _pg_replacement(match: "re.Match[str]") -> str:
    pad = match.group(1)
    return (
        f"{pad}# Connection settings come from the environment (see ragkit/config.py), so a\n"
        f"{pad}# notebook can never reach into whatever happens to be running on port 5432.\n"
        f"{pad}# Override with RAG_PG_HOST / RAG_PG_PORT / RAG_PG_DATABASE.\n"
        f"{pad}POSTGRES_CONFIG = config.POSTGRES_CONFIG"
    )


def env_driven_postgres(nb: dict, path: Path) -> bool:
    """Replace inline POSTGRES_CONFIG dicts with ragkit.config.POSTGRES_CONFIG."""
    changed = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell_source(cell)
        new = PG_CONFIG_RE.sub(_pg_replacement, src)
        if new != src:
            set_source(cell, new)
            changed = True
    return changed



# Notebooks call `ollama.embed(...)` directly, which is exactly what a learner
# should see. Rebinding the name at import time routes those calls through the
# ragkit seam without touching a single call site -- so RAG_FAKE_MODELS=1 works
# for the whole curriculum and the code on screen stays honest.
OLLAMA_SEAM = """# `ollama` below is the real client -- unless RAG_FAKE_MODELS=1 is set, in which
# case it is ragkit.testing.FakeOllamaClient, which implements the same embed()
# and chat() API deterministically. That is what lets CI execute every notebook
# with no model server and no downloads. Every call site stays the same.
from ragkit.embed import get_client

ollama = get_client()"""


def ollama_seam(nb: dict, path: Path) -> bool:
    """Route bare `import ollama` through ragkit.embed.get_client()."""
    changed = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell_source(cell)
        if not re.search(r"^\s*import ollama\s*$", src, re.M):
            continue
        # Module-level import: rebind the name once, with the explanation.
        new = re.sub(r"^import ollama\s*$", OLLAMA_SEAM, src, count=1, flags=re.M)
        # Function-local imports (07-hybrid-search has these) just need the seam.
        new = re.sub(
            r"^([ \t]+)import ollama\s*$",
            r"\1from ragkit.embed import get_client as _get_client\n"
            r"\1ollama = _get_client()  # real client, or the deterministic fake",
            new, flags=re.M,
        )
        if new != src:
            set_source(cell, new)
            changed = True
    return changed


PASSES = {
    "normalize_source": normalize_source,
    "strip_outputs": strip_outputs,
    "add_title": add_title,
    "swap_models": swap_models,
    "add_imports": add_imports,
    "canonical_tables": canonical_tables,
    "remove_input_prompts": remove_input_prompts,
    "dynamic_dimension": dynamic_dimension,
    "env_driven_postgres": env_driven_postgres,
    "ollama_seam": ollama_seam,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("passes", nargs="*", default=["all"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for name, fn in PASSES.items():
            print(f"{name:20s} {(fn.__doc__ or '').strip().splitlines()[0]}")
        return 0

    names = list(PASSES) if args.passes == ["all"] or "all" in args.passes else args.passes
    unknown = [n for n in names if n not in PASSES]
    if unknown:
        print(f"unknown pass(es): {unknown}. Try --list", file=sys.stderr)
        return 2

    total = 0
    for path in notebooks():
        nb = json.loads(path.read_text(encoding="utf-8"))
        touched = [name for name in names if PASSES[name](nb, path)]
        if touched:
            total += 1
            rel = path.relative_to(ROOT)
            print(f"{'would edit' if args.dry_run else 'edited'} {rel}: {', '.join(touched)}")
            if not args.dry_run:
                path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n{total} notebook(s) {'would be ' if args.dry_run else ''}changed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
