"""The teaching copies must agree with the library.

Some duplication in this repository is deliberate. `foundation/01` writes
chunk_text and cosine_similarity out by hand because implementing them *is* the
lesson; every later notebook imports the ragkit version instead. See AGENTS.md.

That arrangement is only safe if something checks the two stay in agreement.
Nothing did before, which is how six textually distinct ndcg_at_k implementations
accumulated -- and how all six ended up sharing the same wrong ideal-DCG.

Each test here extracts the inline definition from its notebook, executes it in
isolation, and compares it to the library function on shared inputs.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def notebook_source(path: Path) -> str:
    nb = json.loads(path.read_text(encoding="utf-8"))
    parts = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", [])
        if isinstance(src, str):
            text = src
        elif src and not any(l.endswith("\n") for l in src[:-1]):
            text = "\n".join(src)
        else:
            text = "".join(src)
        parts.append(text if text.endswith("\n") else text + "\n")
    return "\n".join(parts)


def extract_function(path: Path, name: str):
    """Compile just one top-level function out of a notebook, and return it."""
    source = notebook_source(path)
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            namespace: dict = {}
            preamble = "import math\nimport numpy as np\n"
            exec(preamble + ast.get_source_segment(source, node), namespace)  # noqa: S102
            return namespace[name]
    pytest.fail(f"{path.name} no longer defines {name}() inline. "
                f"If the teaching copy was removed on purpose, remove this test too.")


TEXTS = [
    "Short text under the limit.",
    "Article: Photosynthesis\n\n" + ("Plants convert light into chemical energy. " * 40)
    + "\n\nChlorophyll absorbs light, which is why leaves look green. " * 5,
    "One extremely long paragraph without any breaks. " * 90,
    "",
]

VECTORS = [
    ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
    ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
    ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]),
    ([-1.0, -2.0], [1.0, 2.0]),
    ([0.0, 0.0], [1.0, 1.0]),
]


@pytest.mark.unit
@pytest.mark.parametrize("max_size", [80, 1000])
def test_chunk_text_teaching_copy_matches_library(max_size):
    from ragkit.chunking import chunk_text as library

    inline = extract_function(ROOT / "foundation" / "01-basic-rag-in-memory.ipynb", "chunk_text")
    for text in TEXTS:
        assert inline(text, max_size) == library(text, max_size), (
            f"foundation/01's inline chunk_text disagrees with ragkit.chunking "
            f"on {text[:40]!r} at max_size={max_size}"
        )


@pytest.mark.unit
def test_cosine_similarity_teaching_copy_matches_library():
    from ragkit.retrieval import cosine_similarity as library

    inline = extract_function(
        ROOT / "foundation" / "01-basic-rag-in-memory.ipynb", "cosine_similarity"
    )
    for a, b in VECTORS:
        assert inline(a, b) == pytest.approx(library(a, b), abs=1e-9), (
            f"foundation/01's inline cosine_similarity disagrees with ragkit.retrieval on {a}, {b}"
        )


@pytest.mark.unit
def test_teaching_copies_are_marked():
    """Every inline duplicate must carry the marker AGENTS.md requires.

    The marker is what tells a future reader (or agent) that the duplication is
    intentional and tested, rather than something to tidy away.
    """
    path = ROOT / "foundation" / "01-basic-rag-in-memory.ipynb"
    source = notebook_source(path)
    for name in ("chunk_text", "cosine_similarity"):
        assert f"# TEACHING COPY" in source, (
            f"{path.name} defines {name}() inline without a '# TEACHING COPY' marker"
        )
