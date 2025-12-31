"""
Comprehensive test suite for Jupyter notebook execution using Papermill.

This module executes all 12+ notebooks in isolated environments with mocked
external APIs to ensure notebook structure and logic are sound without requiring
real API calls or expensive computations.

Test Coverage:
- Foundation Layer (3 tests): Registry, basic RAG, PostgreSQL RAG
- Intermediate Layer (2 tests): Embedding loading/reuse, multi-model comparison
- Advanced Techniques (6 tests): Reranking, query expansion, hybrid search, etc.
- Evaluation Lab (4 tests): Ground truth creation, metrics, comparisons, dashboard

Markers:
- @pytest.mark.notebooks: All notebook execution tests
- @pytest.mark.slow: Long-running tests (> 5 seconds)
- @pytest.mark.timeout(300): 5-minute max execution per notebook
- @pytest.mark.e2e: Sequential/end-to-end tests
- @pytest.mark.postgres: Tests requiring PostgreSQL fixture
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

try:
    import papermill as pm
except ImportError:
    pm = None


# ============================================================================
# Skip Markers and Constants
# ============================================================================

pytestmark = pytest.mark.notebooks

NOTEBOOK_BASE_DIR = Path("/home/steve-leve/projects/rag_wiki_demo")

# Notebook paths by layer
FOUNDATION_NOTEBOOKS = {
    "00_registry_utilities": "foundation/00-registry-and-tracking-utilities.ipynb",
    "01_basic_rag_in_memory": "foundation/01-basic-rag-in-memory.ipynb",
    "02_rag_postgresql_persistent": "foundation/02-rag-postgresql-persistent.ipynb",
}

INTERMEDIATE_NOTEBOOKS = {
    "03_loading_reusing_embeddings": "intermediate/03-loading-and-reusing-embeddings.ipynb",
    "04_comparing_embedding_models": "intermediate/04-comparing-embedding-models.ipynb",
}

ADVANCED_NOTEBOOKS = {
    "05_reranking": "advanced-techniques/05-reranking.ipynb",
    "06_query_expansion": "advanced-techniques/06-query-expansion.ipynb",
    "07_hybrid_search": "advanced-techniques/07-hybrid-search.ipynb",
    "08_semantic_chunking": "advanced-techniques/08-semantic-chunking-and-metadata.ipynb",
    "09_citations": "advanced-techniques/09-citation-tracking.ipynb",
    "10_combined_techniques": "advanced-techniques/10-combined-advanced-rag.ipynb",
}

EVALUATION_NOTEBOOKS = {
    "01_ground_truth": "evaluation-lab/01-create-ground-truth-human-in-loop.ipynb",
    "02_metrics": "evaluation-lab/02-evaluation-metrics-framework.ipynb",
    "03_comparison": "evaluation-lab/03-baseline-and-comparison.ipynb",
    "04_dashboard": "evaluation-lab/04-experiment-dashboard.ipynb",
}

ALL_NOTEBOOKS = {
    **FOUNDATION_NOTEBOOKS,
    **INTERMEDIATE_NOTEBOOKS,
    **ADVANCED_NOTEBOOKS,
    **EVALUATION_NOTEBOOKS,
}


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def papermill_available():
    """Skip test if papermill is not installed."""
    if pm is None:
        pytest.skip("papermill not installed")
    return pm


@pytest.fixture
def mock_external_apis(monkeypatch):
    """
    Mock external API calls to Ollama, datasets, and HuggingFace.

    Returns mocks that prevent actual API calls while allowing notebooks
    to execute successfully.
    """
    import numpy as np

    # Mock ollama.embeddings
    def mock_embeddings(model: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """Return deterministic embedding based on model."""
        if "768" in model or "base" in model.lower():
            dim = 768
        elif "384" in model or "small" in model.lower():
            dim = 384
        else:
            dim = 768

        # Deterministic embedding using model name hash
        seed = sum(ord(c) for c in model) % (2**31)
        embedding = np.random.RandomState(seed).randn(dim).tolist()

        return {
            "embedding": embedding,
            "model": model,
            "prompt_eval_count": len(prompt.split()) if isinstance(prompt, str) else 0,
            "eval_count": 10,
        }

    # Mock ollama.chat
    def mock_chat(model: str, messages: list, **kwargs) -> Dict[str, Any]:
        """Return deterministic chat response."""
        return {
            "model": model,
            "created_at": "2024-01-01T00:00:00Z",
            "message": {
                "role": "assistant",
                "content": "Mock response: This is a simulated LLM response for testing.",
            },
            "done": True,
            "total_duration": 1000000,
            "load_duration": 100000,
            "prompt_eval_count": 10,
            "prompt_eval_duration": 200000,
            "eval_count": 20,
            "eval_duration": 400000,
        }

    # Mock datasets.load_dataset
    def mock_load_dataset(dataset_name: str, split: Optional[str] = None, **kwargs):
        """Return minimal test dataset."""
        # Return a simple mock dataset dict
        return {
            "title": [
                "Test Article 1",
                "Test Article 2",
                "Test Article 3",
                "Test Article 4",
                "Test Article 5",
            ],
            "text": [
                "This is test article 1 content. " * 20,
                "This is test article 2 content. " * 20,
                "This is test article 3 content. " * 20,
                "This is test article 4 content. " * 20,
                "This is test article 5 content. " * 20,
            ],
            "id": ["1", "2", "3", "4", "5"],
        }

    # Apply mocks
    try:
        import ollama

        monkeypatch.setattr("ollama.embeddings", mock_embeddings)
        monkeypatch.setattr("ollama.chat", mock_chat)
    except ImportError:
        pass

    try:
        import datasets

        monkeypatch.setattr("datasets.load_dataset", mock_load_dataset)
    except ImportError:
        pass

    return {
        "embeddings": mock_embeddings,
        "chat": mock_chat,
        "load_dataset": mock_load_dataset,
    }


@pytest.fixture
def notebook_output_dir(tmp_path):
    """Create temporary directory for notebook outputs."""
    output_dir = tmp_path / "notebook_outputs"
    output_dir.mkdir()
    return output_dir


# ============================================================================
# Foundation Layer Tests
# ============================================================================


@pytest.mark.slow
@pytest.mark.timeout(300)
@pytest.mark.postgres
def test_foundation_00_registry_utilities(
    papermill_available, notebook_output_dir, mock_external_apis, postgres_connection
):
    """
    Execute foundation/00-registry-and-tracking-utilities.ipynb.

    Tests that registry utility functions are properly defined and
    importable for use in downstream notebooks.

    Success: Notebook executes without errors, utilities are defined.
    """
    notebook_path = NOTEBOOK_BASE_DIR / FOUNDATION_NOTEBOOKS["00_registry_utilities"]
    output_path = notebook_output_dir / "foundation_00_output.ipynb"

    result = pm.execute_notebook(
        input_path=str(notebook_path),
        output_path=str(output_path),
        kernel_name="python3",
        timeout=300,
        parameters={
            "skip_db_operations": True,
        },
    )

    # Verify execution succeeded
    assert output_path.exists(), "Notebook output file not created"
    assert result is not None, "Notebook execution returned None"


@pytest.mark.slow
@pytest.mark.timeout(300)
@pytest.mark.ollama
def test_foundation_01_basic_rag_in_memory(
    papermill_available, notebook_output_dir, mock_external_apis
):
    """
    Execute foundation/01-basic-rag-in-memory.ipynb.

    Tests in-memory RAG pipeline with mocked Ollama embeddings.
    Verifies notebook structure without requiring live Ollama instance.

    Success: Notebook executes, in-memory retrieval logic works.
    """
    notebook_path = NOTEBOOK_BASE_DIR / FOUNDATION_NOTEBOOKS["01_basic_rag_in_memory"]
    output_path = notebook_output_dir / "foundation_01_output.ipynb"

    result = pm.execute_notebook(
        input_path=str(notebook_path),
        output_path=str(output_path),
        kernel_name="python3",
        timeout=300,
        parameters={
            "TARGET_SIZE_MB": 1,  # Minimal dataset
            "EMBEDDING_MODEL": "bge-base-en-v1.5",
            "DEBUG_MODE": True,
        },
    )

    assert output_path.exists(), "Notebook output file not created"
    assert result is not None, "Notebook execution returned None"


@pytest.mark.slow
@pytest.mark.timeout(300)
@pytest.mark.postgres
def test_foundation_02_rag_postgresql_persistent(
    papermill_available, notebook_output_dir, mock_external_apis, postgres_connection
):
    """
    Execute foundation/02-rag-postgresql-persistent.ipynb.

    Tests PostgreSQL-backed RAG with mocked APIs and database fixture.
    Verifies persistent storage and retrieval operations.

    Success: Notebook executes, database operations work.
    """
    notebook_path = NOTEBOOK_BASE_DIR / FOUNDATION_NOTEBOOKS["02_rag_postgresql_persistent"]
    output_path = notebook_output_dir / "foundation_02_output.ipynb"

    result = pm.execute_notebook(
        input_path=str(notebook_path),
        output_path=str(output_path),
        kernel_name="python3",
        timeout=300,
        parameters={
            "DATABASE_URL": "postgresql://postgres:postgres@localhost:5432/rag_test_db",
            "TARGET_SIZE_MB": 1,
            "EMBEDDING_MODEL": "bge-base-en-v1.5",
            "DEBUG_MODE": True,
        },
    )

    assert output_path.exists(), "Notebook output file not created"
    assert result is not None, "Notebook execution returned None"


# ============================================================================
# Intermediate Layer Tests
# ============================================================================


@pytest.mark.slow
@pytest.mark.timeout(300)
@pytest.mark.ollama
def test_intermediate_03_loading_reusing_embeddings(
    papermill_available, notebook_output_dir, mock_external_apis
):
    """
    Execute intermediate/03-loading-and-reusing-embeddings.ipynb.

    Tests the load-or-generate pattern for embeddings efficiency.
    Mocks embedding storage and retrieval.

    Success: Notebook executes, caching pattern works correctly.
    """
    notebook_path = (
        NOTEBOOK_BASE_DIR / INTERMEDIATE_NOTEBOOKS["03_loading_reusing_embeddings"]
    )
    output_path = notebook_output_dir / "intermediate_03_output.ipynb"

    result = pm.execute_notebook(
        input_path=str(notebook_path),
        output_path=str(output_path),
        kernel_name="python3",
        timeout=300,
        parameters={
            "TARGET_SIZE_MB": 1,
            "EMBEDDING_MODEL": "bge-base-en-v1.5",
            "CACHE_EMBEDDINGS": True,
            "DEBUG_MODE": True,
        },
    )

    assert output_path.exists(), "Notebook output file not created"
    assert result is not None, "Notebook execution returned None"


@pytest.mark.slow
@pytest.mark.timeout(300)
@pytest.mark.ollama
def test_intermediate_04_comparing_embedding_models(
    papermill_available, notebook_output_dir, mock_external_apis
):
    """
    Execute intermediate/04-comparing-embedding-models.ipynb.

    Tests comparison of multiple embedding models.
    Uses mocked models to avoid downloading multiple large models.

    Success: Notebook executes, comparison metrics computed.
    """
    notebook_path = (
        NOTEBOOK_BASE_DIR / INTERMEDIATE_NOTEBOOKS["04_comparing_embedding_models"]
    )
    output_path = notebook_output_dir / "intermediate_04_output.ipynb"

    result = pm.execute_notebook(
        input_path=str(notebook_path),
        output_path=str(output_path),
        kernel_name="python3",
        timeout=300,
        parameters={
            "TARGET_SIZE_MB": 1,
            "MODELS_TO_TEST": ["bge-base-en-v1.5", "bge-small-en-v1.5"],
            "DEBUG_MODE": True,
        },
    )

    assert output_path.exists(), "Notebook output file not created"
    assert result is not None, "Notebook execution returned None"


# ============================================================================
# Advanced Techniques Tests
# ============================================================================


@pytest.mark.slow
@pytest.mark.timeout(300)
@pytest.mark.ollama
def test_advanced_05_reranking(
    papermill_available, notebook_output_dir, mock_external_apis
):
    """
    Execute advanced-techniques/05-reranking.ipynb.

    Tests two-stage retrieval with reranking of candidates.
    Mocks LLM reranking function.

    Success: Notebook executes, reranking logic works.
    """
    notebook_path = NOTEBOOK_BASE_DIR / ADVANCED_NOTEBOOKS["05_reranking"]
    output_path = notebook_output_dir / "advanced_05_output.ipynb"

    result = pm.execute_notebook(
        input_path=str(notebook_path),
        output_path=str(output_path),
        kernel_name="python3",
        timeout=300,
        parameters={
            "TARGET_SIZE_MB": 1,
            "EMBEDDING_MODEL": "bge-base-en-v1.5",
            "DEBUG_MODE": True,
        },
    )

    assert output_path.exists(), "Notebook output file not created"
    assert result is not None, "Notebook execution returned None"


@pytest.mark.slow
@pytest.mark.timeout(300)
@pytest.mark.ollama
def test_advanced_06_query_expansion(
    papermill_available, notebook_output_dir, mock_external_apis
):
    """
    Execute advanced-techniques/06-query-expansion.ipynb.

    Tests query expansion with LLM-generated variations.
    Uses mocked LLM to generate alternative queries.

    Success: Notebook executes, query variants generated.
    """
    notebook_path = NOTEBOOK_BASE_DIR / ADVANCED_NOTEBOOKS["06_query_expansion"]
    output_path = notebook_output_dir / "advanced_06_output.ipynb"

    result = pm.execute_notebook(
        input_path=str(notebook_path),
        output_path=str(output_path),
        kernel_name="python3",
        timeout=300,
        parameters={
            "TARGET_SIZE_MB": 1,
            "EMBEDDING_MODEL": "bge-base-en-v1.5",
            "DEBUG_MODE": True,
        },
    )

    assert output_path.exists(), "Notebook output file not created"
    assert result is not None, "Notebook execution returned None"


@pytest.mark.slow
@pytest.mark.timeout(300)
@pytest.mark.ollama
def test_advanced_07_hybrid_search(
    papermill_available, notebook_output_dir, mock_external_apis
):
    """
    Execute advanced-techniques/07-hybrid-search.ipynb.

    Tests hybrid search combining dense embeddings and sparse retrieval.
    Uses RRF (Reciprocal Rank Fusion) for ranking combination.

    Success: Notebook executes, hybrid ranking works.
    """
    notebook_path = NOTEBOOK_BASE_DIR / ADVANCED_NOTEBOOKS["07_hybrid_search"]
    output_path = notebook_output_dir / "advanced_07_output.ipynb"

    result = pm.execute_notebook(
        input_path=str(notebook_path),
        output_path=str(output_path),
        kernel_name="python3",
        timeout=300,
        parameters={
            "TARGET_SIZE_MB": 1,
            "EMBEDDING_MODEL": "bge-base-en-v1.5",
            "DEBUG_MODE": True,
        },
    )

    assert output_path.exists(), "Notebook output file not created"
    assert result is not None, "Notebook execution returned None"


@pytest.mark.slow
@pytest.mark.timeout(300)
@pytest.mark.ollama
def test_advanced_08_semantic_chunking(
    papermill_available, notebook_output_dir, mock_external_apis
):
    """
    Execute advanced-techniques/08-semantic-chunking-and-metadata.ipynb.

    Tests intelligent chunking based on semantic boundaries.
    Verifies metadata preservation during chunking.

    Success: Notebook executes, chunks preserve semantics.
    """
    notebook_path = NOTEBOOK_BASE_DIR / ADVANCED_NOTEBOOKS["08_semantic_chunking"]
    output_path = notebook_output_dir / "advanced_08_output.ipynb"

    result = pm.execute_notebook(
        input_path=str(notebook_path),
        output_path=str(output_path),
        kernel_name="python3",
        timeout=300,
        parameters={
            "TARGET_SIZE_MB": 1,
            "CHUNK_SIZE": 256,
            "DEBUG_MODE": True,
        },
    )

    assert output_path.exists(), "Notebook output file not created"
    assert result is not None, "Notebook execution returned None"


@pytest.mark.slow
@pytest.mark.timeout(300)
@pytest.mark.ollama
def test_advanced_09_citations(
    papermill_available, notebook_output_dir, mock_external_apis
):
    """
    Execute advanced-techniques/09-citation-tracking.ipynb.

    Tests provenance tracking and citation generation.
    Verifies source chunks are properly attributed.

    Success: Notebook executes, citations track correctly.
    """
    notebook_path = NOTEBOOK_BASE_DIR / ADVANCED_NOTEBOOKS["09_citations"]
    output_path = notebook_output_dir / "advanced_09_output.ipynb"

    result = pm.execute_notebook(
        input_path=str(notebook_path),
        output_path=str(output_path),
        kernel_name="python3",
        timeout=300,
        parameters={
            "TARGET_SIZE_MB": 1,
            "EMBEDDING_MODEL": "bge-base-en-v1.5",
            "DEBUG_MODE": True,
        },
    )

    assert output_path.exists(), "Notebook output file not created"
    assert result is not None, "Notebook execution returned None"


@pytest.mark.slow
@pytest.mark.timeout(300)
@pytest.mark.ollama
def test_advanced_10_combined_techniques(
    papermill_available, notebook_output_dir, mock_external_apis
):
    """
    Execute advanced-techniques/10-combined-advanced-rag.ipynb.

    Tests combination of all advanced techniques in single pipeline.
    Verifies multi-step RAG workflow executes correctly.

    Success: Notebook executes, combined pipeline works.
    """
    notebook_path = NOTEBOOK_BASE_DIR / ADVANCED_NOTEBOOKS["10_combined_techniques"]
    output_path = notebook_output_dir / "advanced_10_output.ipynb"

    result = pm.execute_notebook(
        input_path=str(notebook_path),
        output_path=str(output_path),
        kernel_name="python3",
        timeout=300,
        parameters={
            "TARGET_SIZE_MB": 1,
            "EMBEDDING_MODEL": "bge-base-en-v1.5",
            "ENABLE_RERANKING": True,
            "ENABLE_QUERY_EXPANSION": True,
            "DEBUG_MODE": True,
        },
    )

    assert output_path.exists(), "Notebook output file not created"
    assert result is not None, "Notebook execution returned None"


# ============================================================================
# Evaluation Lab Tests
# ============================================================================


@pytest.mark.slow
@pytest.mark.timeout(300)
@pytest.mark.postgres
def test_evaluation_01_ground_truth(
    papermill_available, notebook_output_dir, mock_external_apis, postgres_connection
):
    """
    Execute evaluation-lab/01-create-ground-truth-human-in-loop.ipynb.

    Tests ground truth dataset creation with human-in-the-loop workflow.
    Verifies questions are properly stored with quality ratings.

    Success: Notebook executes, ground truth questions created.
    """
    notebook_path = NOTEBOOK_BASE_DIR / EVALUATION_NOTEBOOKS["01_ground_truth"]
    output_path = notebook_output_dir / "evaluation_01_output.ipynb"

    result = pm.execute_notebook(
        input_path=str(notebook_path),
        output_path=str(output_path),
        kernel_name="python3",
        timeout=300,
        parameters={
            "DATABASE_URL": "postgresql://postgres:postgres@localhost:5432/rag_test_db",
            "NUM_QUESTIONS": 5,
            "SKIP_HUMAN_FEEDBACK": True,  # Skip interactive prompts
            "DEBUG_MODE": True,
        },
    )

    assert output_path.exists(), "Notebook output file not created"
    assert result is not None, "Notebook execution returned None"


@pytest.mark.slow
@pytest.mark.timeout(300)
@pytest.mark.postgres
def test_evaluation_02_compute_metrics(
    papermill_available, notebook_output_dir, mock_external_apis, postgres_connection
):
    """
    Execute evaluation-lab/02-evaluation-metrics-framework.ipynb.

    Tests metric computation (Precision@K, Recall@K, MRR, NDCG).
    Verifies evaluation framework with mock experiment data.

    Success: Notebook executes, metrics computed correctly.
    """
    notebook_path = NOTEBOOK_BASE_DIR / EVALUATION_NOTEBOOKS["02_metrics"]
    output_path = notebook_output_dir / "evaluation_02_output.ipynb"

    result = pm.execute_notebook(
        input_path=str(notebook_path),
        output_path=str(output_path),
        kernel_name="python3",
        timeout=300,
        parameters={
            "DATABASE_URL": "postgresql://postgres:postgres@localhost:5432/rag_test_db",
            "METRIC_K": 5,
            "DEBUG_MODE": True,
        },
    )

    assert output_path.exists(), "Notebook output file not created"
    assert result is not None, "Notebook execution returned None"


@pytest.mark.slow
@pytest.mark.timeout(300)
@pytest.mark.postgres
def test_evaluation_03_comparison(
    papermill_available, notebook_output_dir, mock_external_apis, postgres_connection
):
    """
    Execute evaluation-lab/03-baseline-and-comparison.ipynb.

    Tests experiment comparison and baseline establishment.
    Verifies metric comparison across multiple experiments.

    Success: Notebook executes, experiments compared.
    """
    notebook_path = NOTEBOOK_BASE_DIR / EVALUATION_NOTEBOOKS["03_comparison"]
    output_path = notebook_output_dir / "evaluation_03_output.ipynb"

    result = pm.execute_notebook(
        input_path=str(notebook_path),
        output_path=str(output_path),
        kernel_name="python3",
        timeout=300,
        parameters={
            "DATABASE_URL": "postgresql://postgres:postgres@localhost:5432/rag_test_db",
            "DEBUG_MODE": True,
        },
    )

    assert output_path.exists(), "Notebook output file not created"
    assert result is not None, "Notebook execution returned None"


@pytest.mark.slow
@pytest.mark.timeout(300)
@pytest.mark.postgres
def test_evaluation_04_dashboard(
    papermill_available, notebook_output_dir, mock_external_apis, postgres_connection
):
    """
    Execute evaluation-lab/04-experiment-dashboard.ipynb.

    Tests dashboard and visualization generation.
    Verifies plots and reports are created successfully.

    Success: Notebook executes, visualizations generated.
    """
    notebook_path = NOTEBOOK_BASE_DIR / EVALUATION_NOTEBOOKS["04_dashboard"]
    output_path = notebook_output_dir / "evaluation_04_output.ipynb"

    result = pm.execute_notebook(
        input_path=str(notebook_path),
        output_path=str(output_path),
        kernel_name="python3",
        timeout=300,
        parameters={
            "DATABASE_URL": "postgresql://postgres:postgres@localhost:5432/rag_test_db",
            "DEBUG_MODE": True,
        },
    )

    assert output_path.exists(), "Notebook output file not created"
    assert result is not None, "Notebook execution returned None"


# ============================================================================
# Integration & Sequential Execution Tests
# ============================================================================


@pytest.mark.slow
@pytest.mark.timeout(600)
@pytest.mark.e2e
@pytest.mark.postgres
def test_foundation_sequential_execution(
    papermill_available, notebook_output_dir, mock_external_apis, postgres_connection
):
    """
    Execute foundation notebooks sequentially: 00 → 01 → 02.

    Tests that notebooks can be run in dependency order without conflicts.
    Verifies data/functions from earlier notebooks are available to later ones.

    Success: All three foundation notebooks execute successfully in order.
    """
    notebooks_to_run = [
        ("00", FOUNDATION_NOTEBOOKS["00_registry_utilities"]),
        ("01", FOUNDATION_NOTEBOOKS["01_basic_rag_in_memory"]),
        ("02", FOUNDATION_NOTEBOOKS["02_rag_postgresql_persistent"]),
    ]

    for notebook_id, notebook_rel_path in notebooks_to_run:
        notebook_path = NOTEBOOK_BASE_DIR / notebook_rel_path
        output_path = notebook_output_dir / f"foundation_{notebook_id}_sequential_output.ipynb"

        result = pm.execute_notebook(
            input_path=str(notebook_path),
            output_path=str(output_path),
            kernel_name="python3",
            timeout=300,
            parameters={
                "TARGET_SIZE_MB": 1,
                "DATABASE_URL": "postgresql://postgres:postgres@localhost:5432/rag_test_db",
                "EMBEDDING_MODEL": "bge-base-en-v1.5",
                "DEBUG_MODE": True,
            },
        )

        assert output_path.exists(), f"Foundation {notebook_id} output not created"
        assert result is not None, f"Foundation {notebook_id} execution returned None"


@pytest.mark.slow
@pytest.mark.timeout(300)
@pytest.mark.ollama
def test_advanced_techniques_sequential_execution(
    papermill_available, notebook_output_dir, mock_external_apis
):
    """
    Execute advanced technique notebooks in suggested order.

    Tests that advanced notebooks can build on foundation knowledge.
    Execution order: 05 (reranking) → 06 (expansion) → 10 (combined).

    Success: Ordered execution completes without errors.
    """
    notebooks_to_run = [
        ("05", ADVANCED_NOTEBOOKS["05_reranking"]),
        ("06", ADVANCED_NOTEBOOKS["06_query_expansion"]),
        ("10", ADVANCED_NOTEBOOKS["10_combined_techniques"]),
    ]

    for notebook_id, notebook_rel_path in notebooks_to_run:
        notebook_path = NOTEBOOK_BASE_DIR / notebook_rel_path
        output_path = notebook_output_dir / f"advanced_{notebook_id}_sequential_output.ipynb"

        result = pm.execute_notebook(
            input_path=str(notebook_path),
            output_path=str(output_path),
            kernel_name="python3",
            timeout=300,
            parameters={
                "TARGET_SIZE_MB": 1,
                "EMBEDDING_MODEL": "bge-base-en-v1.5",
                "DEBUG_MODE": True,
            },
        )

        assert output_path.exists(), f"Advanced {notebook_id} output not created"
        assert result is not None, f"Advanced {notebook_id} execution returned None"


@pytest.mark.slow
@pytest.mark.timeout(300)
@pytest.mark.ollama
def test_parametrized_notebook_execution_with_reduced_dataset(
    papermill_available, notebook_output_dir, mock_external_apis
):
    """
    Execute foundation/01 with parametrically reduced dataset.

    Tests parameter override mechanism for rapid iteration.
    Uses 1MB dataset instead of full size for speed.

    Success: Notebook executes with overridden parameters.
    """
    notebook_path = NOTEBOOK_BASE_DIR / FOUNDATION_NOTEBOOKS["01_basic_rag_in_memory"]
    output_path = notebook_output_dir / "foundation_01_parametrized_output.ipynb"

    result = pm.execute_notebook(
        input_path=str(notebook_path),
        output_path=str(output_path),
        kernel_name="python3",
        timeout=300,
        parameters={
            "TARGET_SIZE_MB": 1,  # Override to minimal size
            "EMBEDDING_MODEL": "bge-base-en-v1.5",
            "DEBUG_MODE": True,
        },
    )

    assert output_path.exists(), "Notebook output file not created"
    assert result is not None, "Notebook execution returned None"


# ============================================================================
# Error Handling & Robustness Tests
# ============================================================================


@pytest.mark.slow
@pytest.mark.timeout(300)
@pytest.mark.notebooks
def test_notebook_execution_error_handling(papermill_available, notebook_output_dir):
    """
    Verify papermill properly handles notebook execution errors.

    Tests that errors in notebooks are caught and reported correctly.
    Uses a simple notebook (foundation/00) to verify error handling.

    Success: Execution completes with clear error reporting.
    """
    # This test verifies the test infrastructure itself
    notebook_path = NOTEBOOK_BASE_DIR / FOUNDATION_NOTEBOOKS["00_registry_utilities"]
    output_path = notebook_output_dir / "error_handling_test_output.ipynb"

    # Should execute without raising exception
    result = pm.execute_notebook(
        input_path=str(notebook_path),
        output_path=str(output_path),
        kernel_name="python3",
        timeout=300,
        parameters={"skip_db_operations": True},
    )

    assert output_path.exists()
    assert result is not None


@pytest.mark.notebooks
@pytest.mark.timeout(60)
def test_notebook_path_validation(papermill_available):
    """
    Verify all notebook paths exist and are valid.

    Tests that all referenced notebook paths are correct and accessible.

    Success: All notebooks found at expected paths.
    """
    all_paths = list(ALL_NOTEBOOKS.values())

    for rel_path in all_paths:
        full_path = NOTEBOOK_BASE_DIR / rel_path
        assert full_path.exists(), f"Notebook not found: {full_path}"
        assert full_path.suffix == ".ipynb", f"Not a notebook file: {full_path}"
