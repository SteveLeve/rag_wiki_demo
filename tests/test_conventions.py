"""Guards against the specific bugs this repository actually shipped.

Each test here maps to a defect that existed in main and survived for months
because nothing checked for it. Fixing a bug once is not enough when the same
mistake is spread across six notebooks.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TIERS = ("foundation", "intermediate", "advanced-techniques", "evaluation-lab")
NOTEBOOKS = sorted(p for t in TIERS for p in (ROOT / t).glob("*.ipynb"))


def notebook_code(path: Path) -> str:
    nb = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", [])
        if isinstance(src, str):
            out.append(src)
        elif src and not any(l.endswith("\n") for l in src[:-1]):
            out.append("\n".join(src))
        else:
            out.append("".join(src))
    return "\n".join(out)


@pytest.mark.unit
def test_notebooks_exist():
    assert len(NOTEBOOKS) >= 21, f"expected the full curriculum, found {len(NOTEBOOKS)}"


@pytest.mark.unit
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_no_legacy_ollama_api(path: Path):
    """ollama.embeddings returns non-normalized vectors for nomic-bert models.

    Since this project computes raw cosine similarity, the legacy endpoint gives
    different retrieval results than ollama.embed -- it is not just an old spelling.
    """
    assert "ollama.embeddings" not in notebook_code(path)


@pytest.mark.unit
def test_no_legacy_ollama_api_in_tests():
    """The old mocks patched an API nothing calls, so they never intercepted."""
    import re

    # Match a real patch call, not prose. The docstrings in these files explain
    # the old bug on purpose, and that explanation should not trip the guard.
    call = re.compile(r"""monkeypatch\.setattr\(\s*["']ollama\.embeddings["']""")
    for path in (ROOT / "tests").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert not call.search(source), (
            f"{path.name}: still patches the legacy ollama.embeddings API, "
            "which no caller uses -- patch the ragkit.embed seam instead"
        )


@pytest.mark.unit
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_no_hardcoded_vector_dimension(path: Path):
    """The catalog spans 384/768/1024; a literal vector(768) breaks two of three."""
    import re

    assert not re.search(r"\bvector\(\s*\d+\s*\)", notebook_code(path))


@pytest.mark.unit
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_no_hand_rolled_alias_normalization(path: Path):
    """Six notebooks used .replace('.','_') and a seventh also replaced '-'.

    That split one model across two registry rows and two table names.
    """
    assert '.replace(".", "_")' not in notebook_code(path)
    assert ".replace('.', '_')" not in notebook_code(path)


@pytest.mark.unit
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_no_interactive_input(path: Path):
    """input() blocks forever under papermill, so the notebook can never run in CI."""
    import re

    code = notebook_code(path)
    offenders = [
        line for line in code.splitlines()
        if re.search(r"(?<![\w.])input\s*\(", line) and not line.strip().startswith("#")
    ]
    assert not offenders, f"interactive input() in {path.name}: {offenders[:2]}"


@pytest.mark.unit
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_valid_and_titled(path: Path):
    """Two INDEX notebooks were 0-cell files that crash nbformat and papermill."""
    import nbformat

    nb = nbformat.read(str(path), as_version=4)
    nbformat.validate(nb)
    assert nb.cells, f"{path.name} has no cells"
    first = nb.cells[0]
    assert first.cell_type == "markdown" and first.source.lstrip().startswith("# "), (
        f"{path.name} must open with a markdown H1 title"
    )


@pytest.mark.unit
def test_nb_lint_passes():
    """The linter itself must be green -- it is the CI gate for all of the above."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "nb_lint.py")],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert result.returncode == 0, result.stdout


@pytest.mark.unit
def test_markdown_links_resolve():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_links.py")],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert result.returncode == 0, result.stdout


@pytest.mark.unit
def test_catalog_aliases_are_legal_identifiers():
    """A CHECK constraint enforces this in the database; fail earlier here."""
    import re

    from ragkit.models import EMBEDDING_MODELS

    for alias in EMBEDDING_MODELS:
        assert re.match(r"^[a-z][a-z0-9_]*$", alias), f"{alias} is not a legal SQL identifier"


@pytest.mark.unit
def test_every_notebook_alias_is_in_the_catalog():
    """Eight notebooks once referenced an alias no model backed."""
    import re

    from ragkit.models import EMBEDDING_MODELS, canonical_alias

    known = set(EMBEDDING_MODELS)
    for path in NOTEBOOKS:
        for raw in re.findall(r"EMBEDDING_MODEL_ALIAS\s*=\s*['\"]([^'\"]+)['\"]", notebook_code(path)):
            assert canonical_alias(raw) in known, (
                f"{path.name} references {raw!r}, which is not in the ragkit catalog"
            )


def _create_table_columns(ddl: str, table_marker: str) -> list[str]:
    """Column names from the CREATE TABLE whose body follows `table_marker`."""
    import re

    start = ddl.index(table_marker)
    body = ddl[ddl.index("(", start) + 1:]
    depth, end = 1, 0
    for end, char in enumerate(body):
        depth += (char == "(") - (char == ")")
        if depth == 0:
            break
    names = []
    for line in body[:end].splitlines():
        line = line.strip().rstrip(",")
        if not line or line.startswith(("--", "CONSTRAINT", "PRIMARY KEY", "FOREIGN KEY")):
            continue
        names.append(line.split()[0].lower())
    return names


@pytest.mark.unit
def test_foundation_02_ddl_matches_the_library():
    """foundation/02 writes its own CREATE TABLE; it must not drift from ragkit.

    It did. The notebook's teaching copy omitted the `metadata` column that
    ragkit.db.ensure_embedding_table creates, so whether an embeddings table had
    that column depended on which code path happened to create it -- and
    advanced-techniques/08, whose whole subject is chunk metadata, failed with
    "column metadata does not exist" on a database built by the notebook.
    """
    import inspect

    from ragkit import db

    library = _create_table_columns(
        inspect.getsource(db.ensure_embedding_table), "CREATE TABLE IF NOT EXISTS {table}"
    )
    notebook_ddl = notebook_code(ROOT / "foundation" / "02-rag-postgresql-persistent.ipynb")
    taught = _create_table_columns(notebook_ddl, "CREATE TABLE {self.table_name}")

    assert taught == library, (
        f"foundation/02 creates {taught} but ragkit.db.ensure_embedding_table creates "
        f"{library}; the two must agree or a table's shape depends on who made it"
    )
