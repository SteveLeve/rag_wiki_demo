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


METRICS_NB = ROOT / "evaluation-lab" / "02-evaluation-metrics-framework.ipynb"

# (retrieved, relevant) pairs chosen to separate the two competing ideal-DCG
# readings. The third is the case that settled it: one relevant chunk out of ten,
# ranked first. Normalising against the retrieved set alone scores that a perfect
# 1.000; normalising against everything relevant scores it 0.339.
RANKINGS = [
    ([1, 2, 3, 4, 5], [1, 3, 5]),
    ([1, 2, 3, 4, 5], [2, 4]),
    ([7, 1, 2, 3, 4], [7, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
    ([1, 2, 3], [9, 10]),
    ([], [1, 2]),
    ([1, 2, 3], []),
    ([5, 4, 3, 2, 1], [1, 2, 3, 4, 5]),
]


@pytest.mark.unit
@pytest.mark.parametrize("name", ["precision_at_k", "recall_at_k", "ndcg_at_k"])
@pytest.mark.parametrize("k", [0, 1, 3, 5, 20])
def test_metric_teaching_copies_match_library(name, k):
    from ragkit import metrics

    inline = extract_function(METRICS_NB, name)
    library = getattr(metrics, name)
    for retrieved, relevant in RANKINGS:
        assert inline(retrieved, relevant, k) == pytest.approx(
            library(retrieved, relevant, k), abs=1e-12
        ), (
            f"evaluation-lab/02's inline {name} disagrees with ragkit.metrics on "
            f"retrieved={retrieved}, relevant={relevant}, k={k}"
        )


@pytest.mark.unit
def test_mrr_teaching_copy_matches_library():
    from ragkit.metrics import mean_reciprocal_rank as library

    inline = extract_function(METRICS_NB, "mean_reciprocal_rank")
    for retrieved, relevant in RANKINGS:
        assert inline(retrieved, relevant) == pytest.approx(library(retrieved, relevant))


@pytest.mark.unit
def test_ndcg_teaching_copy_normalises_over_all_relevant():
    """The adjudicated semantics, asserted on the case that distinguishes them.

    This is a guard, not a duplicate of the parity test above: if both the
    notebook and the library regressed together, parity would still pass.
    """
    inline = extract_function(METRICS_NB, "ndcg_at_k")
    one_of_ten_ranked_first = inline([7, 1, 2, 3, 4], list(range(7, 17)), 5)
    assert one_of_ten_ranked_first == pytest.approx(0.339, abs=5e-4)


@pytest.mark.unit
def test_rrf_teaching_copy_matches_library():
    from ragkit.retrieval import reciprocal_rank_fusion as library

    inline = extract_function(
        ROOT / "advanced-techniques" / "07-hybrid-search.ipynb", "reciprocal_rank_fusion"
    )
    cases = [
        ([[1, 2, 3], [3, 2, 1]], 60),
        ([[1, 2, 3], [4, 5, 6]], 60),
        ([[1, 2, 3], [3, 2, 1]], 1),
        ([[], [1]], 60),
        ([[9]], 60),
    ]
    for rankings, k in cases:
        assert inline(rankings, k) == pytest.approx(library(rankings, k)), (
            f"advanced-techniques/07's inline reciprocal_rank_fusion disagrees with "
            f"ragkit.retrieval on {rankings} at k={k}"
        )


# Every notebook that owns a teaching copy, and the names it owns.
MARKED = {
    ("foundation", "01-basic-rag-in-memory"): ["chunk_text", "cosine_similarity"],
    ("evaluation-lab", "02-evaluation-metrics-framework"): [
        "precision_at_k", "recall_at_k", "mean_reciprocal_rank", "ndcg_at_k"],
    ("advanced-techniques", "07-hybrid-search"): [
        "bm25_search_postgresql", "reciprocal_rank_fusion"],
    ("foundation", "00-registry-and-tracking-utilities"): [
        "start_experiment", "complete_experiment", "save_metrics", "compare_experiments"],
}


@pytest.mark.unit
@pytest.mark.parametrize("location,names", sorted(MARKED.items()))
def test_teaching_copies_are_marked(location, names):
    """Every inline duplicate must carry the marker AGENTS.md requires.

    The marker is what tells a future reader (or agent) that the duplication is
    intentional and tested, rather than something to tidy away. scripts/nb_lint.py
    enforces the same rule over the whole tree; this pins the specific set.
    """
    tier, stem = location
    source = notebook_source(ROOT / tier / f"{stem}.ipynb")
    for name in names:
        assert f"def {name}(" in source, f"{tier}/{stem} no longer defines {name}() inline"
    marked = source.count("# TEACHING COPY")
    assert marked >= len(names), (
        f"{tier}/{stem} defines {len(names)} teaching copies but carries only "
        f"{marked} '# TEACHING COPY' marker(s)"
    )
