"""
Unit tests for foundation utility functions.

Tests the registry/tracking utilities from foundation/00 including:
- compute_config_hash: Deterministic configuration hashing
- register_embedding: Embedding model registration
- list_available_embeddings: Registry discovery
- get_embedding_metadata: Metadata retrieval
- start_experiment: Experiment lifecycle start
- complete_experiment: Experiment completion
- save_metrics: Metrics storage (DB + JSON)
- list_experiments: Experiment querying
- get_experiment: Experiment retrieval
- compare_experiments: Experiment comparison

Coverage target: 100% of utility functions
"""

import json
import hashlib
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import psycopg2
import psycopg2.extras
import pytest


# ============================================================================
# Utility Functions (extracted from foundation/00 notebook)
# ============================================================================


def compute_config_hash(config_dict: Dict) -> str:
    """Create deterministic SHA256 hash of a configuration dictionary.

    Args:
        config_dict: Configuration parameters

    Returns:
        SHA256 hash string (first 12 characters for readability)
    """
    config_str = json.dumps(config_dict, sort_keys=True)
    hash_obj = hashlib.sha256(config_str.encode())
    return hash_obj.hexdigest()[:12]


def register_embedding(db_connection, model_alias: str, model_name: str,
                       dimension: int, embedding_count: int,
                       chunk_source_dataset: str = None,
                       chunk_size_config: int = None,
                       metadata: Dict = None) -> bool:
    """Register or update an embedding model in the registry.

    Args:
        db_connection: PostgreSQL connection
        model_alias: Identifier for the model
        model_name: Human-readable model name
        dimension: Embedding vector dimension
        embedding_count: Number of embeddings stored
        chunk_source_dataset: Description of source data
        chunk_size_config: MAX_CHUNK_SIZE used during generation
        metadata: Optional dict with additional info

    Returns:
        True if successful
    """
    if metadata is None:
        metadata = {}

    try:
        with db_connection.cursor() as cur:
            cur.execute('''
                INSERT INTO embedding_registry (
                    model_alias, model_name, dimension, embedding_count,
                    chunk_source_dataset, chunk_size_config, metadata_json
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (model_alias) DO UPDATE SET
                    embedding_count = EXCLUDED.embedding_count,
                    chunk_source_dataset = COALESCE(EXCLUDED.chunk_source_dataset, embedding_registry.chunk_source_dataset),
                    chunk_size_config = COALESCE(EXCLUDED.chunk_size_config, embedding_registry.chunk_size_config),
                    metadata_json = EXCLUDED.metadata_json,
                    last_accessed = CURRENT_TIMESTAMP
            ''', (
                model_alias, model_name, dimension, embedding_count,
                chunk_source_dataset, chunk_size_config, json.dumps(metadata)
            ))
        db_connection.commit()
        return True
    except Exception as e:
        db_connection.rollback()
        raise


def list_available_embeddings(db_connection) -> pd.DataFrame:
    """Query embedding_registry to show available models with metadata.

    Returns:
        DataFrame with columns: model_alias, model_name, dimension, embedding_count,
                                 chunk_source_dataset, created_at, chunk_size_config
    """
    query = '''
        SELECT
            model_alias,
            model_name,
            dimension,
            embedding_count,
            chunk_source_dataset,
            chunk_size_config,
            created_at,
            last_accessed
        FROM embedding_registry
        ORDER BY created_at DESC
    '''
    return pd.read_sql(query, db_connection)


def get_embedding_metadata(db_connection, model_alias: str) -> Optional[Dict]:
    """Fetch metadata_json and other info for a specific model.

    Args:
        db_connection: PostgreSQL connection
        model_alias: The model alias

    Returns:
        Dict with metadata or None if not found
    """
    with db_connection.cursor() as cur:
        cur.execute('''
            SELECT
                dimension,
                embedding_count,
                chunk_source_dataset,
                chunk_size_config,
                created_at,
                metadata_json
            FROM embedding_registry
            WHERE model_alias = %s
        ''', (model_alias,))
        result = cur.fetchone()

        if not result:
            return None

        return {
            'dimension': result[0],
            'embedding_count': result[1],
            'chunk_source_dataset': result[2],
            'chunk_size_config': result[3],
            'created_at': result[4],
            'metadata_json': result[5] or {}
        }


def start_experiment(db_connection, experiment_name: str,
                     notebook_path: str = None,
                     embedding_model_alias: str = None,
                     config: Dict = None,
                     techniques: list = None,
                     notes: str = None) -> int:
    """Start a new experiment and return its ID for tracking.

    Args:
        db_connection: PostgreSQL connection
        experiment_name: Human-readable experiment name
        notebook_path: Path to the notebook running this experiment
        embedding_model_alias: Which embedding model is being used
        config: Dict of configuration parameters
        techniques: List of techniques being applied
        notes: Optional notes about the experiment

    Returns:
        Experiment ID for use in save_metrics() and complete_experiment()
    """
    if config is None:
        config = {}
    if techniques is None:
        techniques = []

    config_hash = compute_config_hash(config)

    with db_connection.cursor() as cur:
        cur.execute('''
            INSERT INTO experiments (
                experiment_name, notebook_path, embedding_model_alias,
                config_hash, config_json, techniques_applied, notes, status
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, 'running')
            RETURNING id
        ''', (
            experiment_name,
            notebook_path,
            embedding_model_alias,
            config_hash,
            json.dumps(config),
            techniques,
            notes
        ))
        exp_id = cur.fetchone()[0]
    db_connection.commit()
    return exp_id


def complete_experiment(db_connection, experiment_id: int,
                       status: str = 'completed',
                       notes: str = None) -> bool:
    """Mark an experiment as complete.

    Args:
        db_connection: PostgreSQL connection
        experiment_id: ID returned from start_experiment()
        status: 'completed' or 'failed'
        notes: Optional update to notes field

    Returns:
        True if successful
    """
    try:
        with db_connection.cursor() as cur:
            update_notes = ", notes = %s" if notes else ""
            params = [status, experiment_id] if not notes else [status, notes, experiment_id]

            cur.execute(f'''
                UPDATE experiments
                SET status = %s{update_notes}, completed_at = CURRENT_TIMESTAMP
                WHERE id = %s
            ''', params)
        db_connection.commit()
        return True
    except Exception as e:
        db_connection.rollback()
        raise


def save_metrics(db_connection, experiment_id: int, metrics_dict: Dict,
                 export_to_file: bool = True,
                 export_dir: str = 'data/experiment_results') -> tuple:
    """Save experiment metrics to database and optionally to JSON file.

    Args:
        db_connection: PostgreSQL connection
        experiment_id: ID from start_experiment()
        metrics_dict: Dict of {metric_name: value, ...}
        export_to_file: Whether to also save to filesystem JSON
        export_dir: Directory for JSON exports

    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        with db_connection.cursor() as cur:
            for metric_name, metric_data in metrics_dict.items():
                # Handle both simple floats and nested dicts with details
                if isinstance(metric_data, dict):
                    metric_value = metric_data.get('value', 0.0)
                    metric_details = metric_data.get('details', {})
                else:
                    metric_value = metric_data
                    metric_details = {}

                cur.execute('''
                    INSERT INTO evaluation_results (
                        experiment_id, metric_name, metric_value, metric_details_json
                    )
                    VALUES (%s, %s, %s, %s)
                ''', (
                    experiment_id,
                    metric_name,
                    float(metric_value),
                    json.dumps(metric_details) if metric_details else '{}'
                ))
        db_connection.commit()

        # Export to file if requested
        file_path = None
        if export_to_file:
            os.makedirs(export_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_path = os.path.join(export_dir, f'experiment_{experiment_id}_{timestamp}.json')
            with open(file_path, 'w') as f:
                json.dump({
                    'experiment_id': experiment_id,
                    'timestamp': timestamp,
                    'metrics': metrics_dict
                }, f, indent=2)

        msg = f"Saved {len(metrics_dict)} metrics for experiment #{experiment_id}"
        if file_path:
            msg += f" to {file_path}"
        return True, msg
    except Exception as e:
        db_connection.rollback()
        return False, str(e)


def get_experiment(db_connection, experiment_id: int) -> Optional[Dict]:
    """Fetch experiment details and associated metrics.

    Args:
        db_connection: PostgreSQL connection
        experiment_id: Experiment ID

    Returns:
        Dict with experiment info and metrics
    """
    with db_connection.cursor() as cur:
        # Get experiment
        cur.execute('SELECT * FROM experiments WHERE id = %s', (experiment_id,))
        exp = cur.fetchone()

        if not exp:
            return None

        # Get metrics for this experiment
        cur.execute('''
            SELECT metric_name, metric_value, metric_details_json
            FROM evaluation_results
            WHERE experiment_id = %s
            ORDER BY metric_name
        ''', (experiment_id,))
        metrics = {row[0]: {'value': row[1], 'details': row[2]} for row in cur.fetchall()}

        return {
            'id': exp[0],
            'name': exp[1],
            'notebook': exp[2],
            'embedding_model': exp[3],
            'config_hash': exp[4],
            'config': exp[5],
            'techniques': exp[6],
            'started_at': exp[7],
            'completed_at': exp[8],
            'status': exp[9],
            'notes': exp[10],
            'metrics': metrics
        }


def list_experiments(db_connection, limit: int = 20,
                    status: str = None,
                    embedding_model: str = None) -> pd.DataFrame:
    """List recent experiments with optional filtering.

    Args:
        db_connection: PostgreSQL connection
        limit: Max number of results
        status: Filter by status ('running', 'completed', 'failed')
        embedding_model: Filter by embedding model alias

    Returns:
        DataFrame of experiments
    """
    query = 'SELECT * FROM experiments WHERE 1=1'
    params = []

    if status:
        query += ' AND status = %s'
        params.append(status)

    if embedding_model:
        query += ' AND embedding_model_alias = %s'
        params.append(embedding_model)

    query += f' ORDER BY started_at DESC LIMIT {limit}'

    return pd.read_sql(query, db_connection, params=params)


def compare_experiments(db_connection, experiment_ids: list,
                       metric_names: list = None) -> pd.DataFrame:
    """Compare metrics across multiple experiments side-by-side.

    Args:
        db_connection: PostgreSQL connection
        experiment_ids: List of experiment IDs to compare
        metric_names: Specific metrics to compare (if None, all metrics)

    Returns:
        DataFrame with experiments as rows, metrics as columns
    """
    if not experiment_ids:
        return pd.DataFrame()

    placeholders = ','.join(['%s'] * len(experiment_ids))

    query = f'''
        SELECT
            e.id,
            e.experiment_name,
            e.embedding_model_alias,
            r.metric_name,
            r.metric_value
        FROM experiments e
        LEFT JOIN evaluation_results r ON e.id = r.experiment_id
        WHERE e.id IN ({placeholders})
    '''

    if metric_names:
        placeholders_metrics = ','.join(['%s'] * len(metric_names))
        query += f' AND r.metric_name IN ({placeholders_metrics})'
        params = experiment_ids + metric_names
    else:
        params = experiment_ids

    df = pd.read_sql(query, db_connection, params=params)

    if df.empty:
        return df

    # Pivot to get metrics as columns
    return df.pivot_table(
        index=['id', 'experiment_name', 'embedding_model_alias'],
        columns='metric_name',
        values='metric_value'
    ).reset_index()


# ============================================================================
# Test Classes and Methods
# ============================================================================


class TestComputeConfigHash:
    """Tests for compute_config_hash() function."""

    @pytest.mark.unit
    def test_deterministic_hash_same_config(self):
        """Same config should always produce same hash."""
        config = {'learning_rate': 0.001, 'batch_size': 32, 'epochs': 10}
        hash1 = compute_config_hash(config)
        hash2 = compute_config_hash(config)
        assert hash1 == hash2

    @pytest.mark.unit
    def test_different_configs_different_hash(self):
        """Different configs should produce different hashes."""
        config1 = {'learning_rate': 0.001, 'batch_size': 32}
        config2 = {'learning_rate': 0.002, 'batch_size': 32}
        hash1 = compute_config_hash(config1)
        hash2 = compute_config_hash(config2)
        assert hash1 != hash2

    @pytest.mark.unit
    def test_hash_length(self):
        """Hash should be 12 characters (SHA256[:12])."""
        config = {'key': 'value'}
        hash_result = compute_config_hash(config)
        assert len(hash_result) == 12

    @pytest.mark.unit
    def test_hash_is_hexadecimal(self):
        """Hash should be valid hexadecimal string."""
        config = {'test': 'data'}
        hash_result = compute_config_hash(config)
        # Should not raise ValueError
        int(hash_result, 16)

    @pytest.mark.unit
    def test_key_order_irrelevant(self):
        """Key ordering in config dict should not affect hash."""
        config1 = {'a': 1, 'b': 2, 'c': 3}
        config2 = {'c': 3, 'a': 1, 'b': 2}
        hash1 = compute_config_hash(config1)
        hash2 = compute_config_hash(config2)
        assert hash1 == hash2

    @pytest.mark.unit
    def test_empty_config(self):
        """Empty config should still produce valid hash."""
        config = {}
        hash_result = compute_config_hash(config)
        assert len(hash_result) == 12
        assert isinstance(hash_result, str)

    @pytest.mark.unit
    def test_nested_config(self):
        """Nested config dicts should hash correctly."""
        config = {'model': {'name': 'bert', 'size': 'base'}, 'lr': 0.001}
        hash_result = compute_config_hash(config)
        assert len(hash_result) == 12


class TestRegisterEmbedding:
    """Tests for register_embedding() function."""

    @pytest.mark.postgres
    def test_new_registration(self, postgres_connection):
        """Registering new embedding model should succeed."""
        result = register_embedding(
            postgres_connection,
            'test_model_1',
            'Test Model 1',
            768,
            1000,
            chunk_source_dataset='Test Dataset',
            chunk_size_config=512
        )
        assert result is True

        # Verify it was inserted
        with postgres_connection.cursor() as cur:
            cur.execute('SELECT * FROM embedding_registry WHERE model_alias = %s', ('test_model_1',))
            row = cur.fetchone()
            assert row is not None
            assert row[1] == 'test_model_1'
            assert row[3] == 768
            assert row[4] == 1000

    @pytest.mark.postgres
    def test_duplicate_alias_upsert(self, postgres_connection):
        """Duplicate alias should update existing record (upsert)."""
        # Insert first time
        register_embedding(postgres_connection, 'dup_model', 'Model V1', 768, 100)

        # Insert again with updated count
        result = register_embedding(postgres_connection, 'dup_model', 'Model V2', 768, 200)
        assert result is True

        # Verify count was updated
        with postgres_connection.cursor() as cur:
            cur.execute('SELECT embedding_count FROM embedding_registry WHERE model_alias = %s', ('dup_model',))
            count = cur.fetchone()[0]
            assert count == 200

    @pytest.mark.postgres
    def test_metadata_json_storage(self, postgres_connection):
        """Custom metadata should be stored correctly."""
        metadata = {
            'training_date': '2024-01-15',
            'license': 'MIT',
            'huggingface_url': 'https://huggingface.co/model'
        }
        register_embedding(
            postgres_connection,
            'meta_model',
            'Model with Metadata',
            384,
            500,
            metadata=metadata
        )

        # Verify metadata
        with postgres_connection.cursor() as cur:
            cur.execute('SELECT metadata_json FROM embedding_registry WHERE model_alias = %s', ('meta_model',))
            stored_meta = json.loads(cur.fetchone()[0])
            assert stored_meta == metadata

    @pytest.mark.postgres
    def test_default_metadata_empty(self, postgres_connection):
        """When no metadata provided, should store empty dict."""
        register_embedding(
            postgres_connection,
            'no_meta_model',
            'Simple Model',
            768,
            100
        )

        with postgres_connection.cursor() as cur:
            cur.execute('SELECT metadata_json FROM embedding_registry WHERE model_alias = %s', ('no_meta_model',))
            meta = json.loads(cur.fetchone()[0])
            assert meta == {}

    @pytest.mark.postgres
    def test_null_optional_fields(self, postgres_connection):
        """Optional fields can be NULL."""
        register_embedding(
            postgres_connection,
            'minimal_model',
            'Minimal Model',
            128,
            50
        )

        with postgres_connection.cursor() as cur:
            cur.execute('SELECT chunk_source_dataset, chunk_size_config FROM embedding_registry WHERE model_alias = %s', ('minimal_model',))
            dataset, chunk_size = cur.fetchone()
            assert dataset is None
            assert chunk_size is None

    @pytest.mark.postgres
    def test_large_embedding_count(self, postgres_connection):
        """Should handle large embedding counts."""
        large_count = 1_000_000
        register_embedding(
            postgres_connection,
            'large_model',
            'Large Dataset Model',
            768,
            large_count
        )

        with postgres_connection.cursor() as cur:
            cur.execute('SELECT embedding_count FROM embedding_registry WHERE model_alias = %s', ('large_model',))
            count = cur.fetchone()[0]
            assert count == large_count


class TestListAvailableEmbeddings:
    """Tests for list_available_embeddings() function."""

    @pytest.mark.postgres
    def test_empty_registry(self, postgres_connection):
        """Empty registry should return empty DataFrame."""
        df = list_available_embeddings(postgres_connection)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    @pytest.mark.postgres
    def test_with_seed_data(self, postgres_connection, seed_test_data):
        """Should return all registered models."""
        df = list_available_embeddings(postgres_connection)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert 'model_alias' in df.columns
        assert 'dimension' in df.columns

    @pytest.mark.postgres
    def test_correct_columns_returned(self, postgres_connection, seed_test_data):
        """Should return all expected columns."""
        df = list_available_embeddings(postgres_connection)
        expected_cols = ['model_alias', 'model_name', 'dimension', 'embedding_count',
                        'chunk_source_dataset', 'chunk_size_config', 'created_at', 'last_accessed']
        for col in expected_cols:
            assert col in df.columns

    @pytest.mark.postgres
    def test_ordering_by_created_date(self, postgres_connection):
        """Results should be ordered by created_at DESC."""
        # Register models with small delay
        register_embedding(postgres_connection, 'first', 'First', 768, 100)
        register_embedding(postgres_connection, 'second', 'Second', 768, 100)
        register_embedding(postgres_connection, 'third', 'Third', 768, 100)

        df = list_available_embeddings(postgres_connection)
        assert df.iloc[0]['model_alias'] == 'third'
        assert df.iloc[-1]['model_alias'] == 'first'

    @pytest.mark.postgres
    def test_multiple_registrations(self, postgres_connection):
        """Should list multiple distinct models."""
        for i in range(5):
            register_embedding(postgres_connection, f'model_{i}', f'Model {i}', 768, 100 * (i + 1))

        df = list_available_embeddings(postgres_connection)
        assert len(df) == 5
        assert all(f'model_{i}' in df['model_alias'].values for i in range(5))


class TestGetEmbeddingMetadata:
    """Tests for get_embedding_metadata() function."""

    @pytest.mark.postgres
    def test_known_alias(self, postgres_connection, seed_test_data):
        """Should return metadata for known model."""
        metadata = get_embedding_metadata(postgres_connection, 'bge_base_en_v1_5')
        assert metadata is not None
        assert metadata['dimension'] == 768
        assert metadata['embedding_count'] == 1000

    @pytest.mark.postgres
    def test_unknown_alias(self, postgres_connection):
        """Unknown alias should return None."""
        metadata = get_embedding_metadata(postgres_connection, 'nonexistent_model')
        assert metadata is None

    @pytest.mark.postgres
    def test_metadata_json_parsed(self, postgres_connection):
        """metadata_json should be included in result."""
        metadata = {'author': 'test', 'version': '1.0'}
        register_embedding(postgres_connection, 'test_meta', 'Test', 768, 100, metadata=metadata)

        result = get_embedding_metadata(postgres_connection, 'test_meta')
        assert result['metadata_json'] == metadata

    @pytest.mark.postgres
    def test_all_fields_present(self, postgres_connection, seed_test_data):
        """Result should have all expected fields."""
        metadata = get_embedding_metadata(postgres_connection, 'bge_small_en_v1_5')
        assert 'dimension' in metadata
        assert 'embedding_count' in metadata
        assert 'chunk_source_dataset' in metadata
        assert 'chunk_size_config' in metadata
        assert 'created_at' in metadata
        assert 'metadata_json' in metadata

    @pytest.mark.postgres
    def test_chunk_size_config_access(self, postgres_connection):
        """Should access chunk_size_config if set."""
        register_embedding(postgres_connection, 'chunked', 'Model', 768, 100,
                          chunk_size_config=1024)

        metadata = get_embedding_metadata(postgres_connection, 'chunked')
        assert metadata['chunk_size_config'] == 1024

    @pytest.mark.postgres
    def test_created_at_timestamp(self, postgres_connection):
        """created_at should be a valid timestamp."""
        register_embedding(postgres_connection, 'timed', 'Model', 768, 100)
        metadata = get_embedding_metadata(postgres_connection, 'timed')
        assert metadata['created_at'] is not None
        assert isinstance(metadata['created_at'], datetime)


class TestExperimentLifecycle:
    """Tests for start_experiment(), complete_experiment()."""

    @pytest.mark.postgres
    def test_start_experiment_returns_id(self, postgres_connection, seed_test_data):
        """start_experiment should return valid experiment ID."""
        exp_id = start_experiment(postgres_connection, 'Test Experiment 1')
        assert isinstance(exp_id, int)
        assert exp_id > 0

    @pytest.mark.postgres
    def test_default_status_running(self, postgres_connection, seed_test_data):
        """New experiment should have status='running'."""
        exp_id = start_experiment(postgres_connection, 'Test Experiment 2')

        with postgres_connection.cursor() as cur:
            cur.execute('SELECT status FROM experiments WHERE id = %s', (exp_id,))
            status = cur.fetchone()[0]
            assert status == 'running'

    @pytest.mark.postgres
    def test_config_hash_computed(self, postgres_connection, seed_test_data):
        """Config hash should be computed and stored."""
        config = {'lr': 0.001, 'batch_size': 32}
        exp_id = start_experiment(postgres_connection, 'Test', config=config)

        with postgres_connection.cursor() as cur:
            cur.execute('SELECT config_hash FROM experiments WHERE id = %s', (exp_id,))
            hash_result = cur.fetchone()[0]
            assert hash_result == compute_config_hash(config)

    @pytest.mark.postgres
    def test_complete_experiment_status_change(self, postgres_connection, seed_test_data):
        """complete_experiment should change status to 'completed'."""
        exp_id = start_experiment(postgres_connection, 'Test')
        result = complete_experiment(postgres_connection, exp_id)
        assert result is True

        with postgres_connection.cursor() as cur:
            cur.execute('SELECT status FROM experiments WHERE id = %s', (exp_id,))
            status = cur.fetchone()[0]
            assert status == 'completed'

    @pytest.mark.postgres
    def test_completed_at_timestamp(self, postgres_connection, seed_test_data):
        """completed_at timestamp should be set."""
        exp_id = start_experiment(postgres_connection, 'Test')
        complete_experiment(postgres_connection, exp_id)

        with postgres_connection.cursor() as cur:
            cur.execute('SELECT completed_at FROM experiments WHERE id = %s', (exp_id,))
            completed_at = cur.fetchone()[0]
            assert completed_at is not None
            assert isinstance(completed_at, datetime)

    @pytest.mark.postgres
    def test_idempotent_completion(self, postgres_connection, seed_test_data):
        """Completing twice should be safe (idempotent)."""
        exp_id = start_experiment(postgres_connection, 'Test')
        result1 = complete_experiment(postgres_connection, exp_id)
        result2 = complete_experiment(postgres_connection, exp_id)
        assert result1 is True
        assert result2 is True

    @pytest.mark.postgres
    def test_complete_with_failed_status(self, postgres_connection, seed_test_data):
        """Should be able to mark experiment as failed."""
        exp_id = start_experiment(postgres_connection, 'Test')
        result = complete_experiment(postgres_connection, exp_id, status='failed')
        assert result is True

        with postgres_connection.cursor() as cur:
            cur.execute('SELECT status FROM experiments WHERE id = %s', (exp_id,))
            status = cur.fetchone()[0]
            assert status == 'failed'

    @pytest.mark.postgres
    def test_start_with_all_parameters(self, postgres_connection, seed_test_data):
        """Should handle all optional parameters."""
        exp_id = start_experiment(
            postgres_connection,
            'Full Test',
            notebook_path='foundation/02.ipynb',
            embedding_model_alias='bge_base_en_v1_5',
            config={'lr': 0.001},
            techniques=['reranking', 'query_expansion'],
            notes='This is a test'
        )
        assert isinstance(exp_id, int)

        with postgres_connection.cursor() as cur:
            cur.execute('SELECT * FROM experiments WHERE id = %s', (exp_id,))
            row = cur.fetchone()
            assert row[1] == 'Full Test'  # experiment_name
            assert 'foundation/02.ipynb' in (row[2] or '')

    @pytest.mark.postgres
    def test_config_json_stored(self, postgres_connection, seed_test_data):
        """Configuration should be stored as JSON."""
        config = {'param1': 'value1', 'param2': 42, 'param3': [1, 2, 3]}
        exp_id = start_experiment(postgres_connection, 'Test', config=config)

        with postgres_connection.cursor() as cur:
            cur.execute('SELECT config_json FROM experiments WHERE id = %s', (exp_id,))
            stored_config = json.loads(cur.fetchone()[0])
            assert stored_config == config


class TestSaveMetrics:
    """Tests for save_metrics() function."""

    @pytest.mark.postgres
    def test_insert_single_metric(self, postgres_connection, seed_test_data):
        """Should insert single metric correctly."""
        exp_id = start_experiment(postgres_connection, 'Test')
        success, msg = save_metrics(postgres_connection, exp_id, {'accuracy': 0.95}, export_to_file=False)
        assert success is True

        with postgres_connection.cursor() as cur:
            cur.fetchall()
            cur.execute('SELECT metric_name, metric_value FROM evaluation_results WHERE experiment_id = %s', (exp_id,))
            rows = cur.fetchall()
            assert len(rows) == 1
            assert rows[0][0] == 'accuracy'
            assert rows[0][1] == 0.95

    @pytest.mark.postgres
    def test_insert_multiple_metrics(self, postgres_connection, seed_test_data):
        """Should insert multiple metrics for one experiment."""
        exp_id = start_experiment(postgres_connection, 'Test')
        metrics = {'accuracy': 0.95, 'precision': 0.92, 'recall': 0.88, 'f1': 0.90}
        success, msg = save_metrics(postgres_connection, exp_id, metrics, export_to_file=False)
        assert success is True

        with postgres_connection.cursor() as cur:
            cur.fetchall()
            cur.execute('SELECT COUNT(*) FROM evaluation_results WHERE experiment_id = %s', (exp_id,))
            count = cur.fetchone()[0]
            assert count == 4

    @pytest.mark.postgres
    def test_dual_output_db_and_json(self, postgres_connection, seed_test_data):
        """Should save to both DB and JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_id = start_experiment(postgres_connection, 'Test')
            metrics = {'accuracy': 0.95}
            success, msg = save_metrics(postgres_connection, exp_id, metrics,
                                       export_to_file=True, export_dir=tmpdir)
            assert success is True

            # Check file was created
            files = os.listdir(tmpdir)
            assert len(files) == 1
            assert files[0].startswith(f'experiment_{exp_id}')

            # Check file contents
            with open(os.path.join(tmpdir, files[0])) as f:
                data = json.load(f)
                assert data['experiment_id'] == exp_id
                assert data['metrics'] == metrics

    @pytest.mark.postgres
    def test_metric_nested_dict_with_details(self, postgres_connection, seed_test_data):
        """Should handle nested dict with value and details."""
        exp_id = start_experiment(postgres_connection, 'Test')
        metrics = {
            'accuracy': {'value': 0.95, 'details': {'true_positives': 95, 'total': 100}},
            'precision': 0.92
        }
        success, msg = save_metrics(postgres_connection, exp_id, metrics, export_to_file=False)
        assert success is True

        with postgres_connection.cursor() as cur:
            cur.fetchall()
            cur.execute('SELECT metric_details_json FROM evaluation_results WHERE metric_name = %s', ('accuracy',))
            details = json.loads(cur.fetchone()[0])
            assert details['true_positives'] == 95

    @pytest.mark.postgres
    def test_no_file_export_when_disabled(self, postgres_connection, seed_test_data):
        """Should not create file when export_to_file=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_id = start_experiment(postgres_connection, 'Test')
            success, msg = save_metrics(postgres_connection, exp_id, {'accuracy': 0.95},
                                       export_to_file=False, export_dir=tmpdir)
            assert success is True

            files = os.listdir(tmpdir)
            assert len(files) == 0

    @pytest.mark.postgres
    def test_message_contains_metric_count(self, postgres_connection, seed_test_data):
        """Success message should mention metric count."""
        exp_id = start_experiment(postgres_connection, 'Test')
        metrics = {'m1': 0.1, 'm2': 0.2, 'm3': 0.3}
        success, msg = save_metrics(postgres_connection, exp_id, metrics, export_to_file=False)
        assert '3' in msg


class TestListExperiments:
    """Tests for list_experiments() function."""

    @pytest.mark.postgres
    def test_query_all_experiments(self, postgres_connection, seed_test_data):
        """Should return all experiments."""
        # Create 3 experiments
        for i in range(3):
            start_experiment(postgres_connection, f'Exp {i}')

        df = list_experiments(postgres_connection, limit=100)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    @pytest.mark.postgres
    def test_filter_by_status_running(self, postgres_connection, seed_test_data):
        """Should filter by status='running'."""
        id1 = start_experiment(postgres_connection, 'Running')
        id2 = start_experiment(postgres_connection, 'ToComplete')
        complete_experiment(postgres_connection, id2)

        df = list_experiments(postgres_connection, limit=100, status='running')
        assert len(df) == 1
        assert df.iloc[0]['status'] == 'running'

    @pytest.mark.postgres
    def test_filter_by_status_completed(self, postgres_connection, seed_test_data):
        """Should filter by status='completed'."""
        id1 = start_experiment(postgres_connection, 'ToComplete')
        complete_experiment(postgres_connection, id1)

        df = list_experiments(postgres_connection, limit=100, status='completed')
        assert len(df) == 1
        assert df.iloc[0]['status'] == 'completed'

    @pytest.mark.postgres
    def test_filter_by_embedding_model(self, postgres_connection, seed_test_data):
        """Should filter by embedding model alias."""
        start_experiment(postgres_connection, 'Exp1', embedding_model_alias='bge_base_en_v1_5')
        start_experiment(postgres_connection, 'Exp2', embedding_model_alias='bge_small_en_v1_5')

        df = list_experiments(postgres_connection, limit=100, embedding_model='bge_base_en_v1_5')
        assert len(df) == 1
        assert df.iloc[0]['embedding_model_alias'] == 'bge_base_en_v1_5'

    @pytest.mark.postgres
    def test_limit_parameter(self, postgres_connection, seed_test_data):
        """Should respect limit parameter."""
        for i in range(10):
            start_experiment(postgres_connection, f'Exp {i}')

        df = list_experiments(postgres_connection, limit=5)
        assert len(df) <= 5

    @pytest.mark.postgres
    def test_sort_by_date_descending(self, postgres_connection, seed_test_data):
        """Should sort by started_at DESC."""
        exp_ids = []
        for i in range(3):
            exp_ids.append(start_experiment(postgres_connection, f'Exp {i}'))

        df = list_experiments(postgres_connection, limit=100)
        # Most recent should be first
        assert df.iloc[0]['id'] == exp_ids[-1]


class TestGetExperiment:
    """Tests for get_experiment() function."""

    @pytest.mark.postgres
    def test_existing_id_returns_record(self, postgres_connection, seed_test_data):
        """Should return experiment record for valid ID."""
        exp_id = start_experiment(postgres_connection, 'Test Exp', notebook_path='test.ipynb')
        result = get_experiment(postgres_connection, exp_id)
        assert result is not None
        assert result['id'] == exp_id
        assert result['name'] == 'Test Exp'

    @pytest.mark.postgres
    def test_non_existent_id_returns_none(self, postgres_connection):
        """Non-existent ID should return None."""
        result = get_experiment(postgres_connection, 9999)
        assert result is None

    @pytest.mark.postgres
    def test_config_json_parsed(self, postgres_connection, seed_test_data):
        """config_json should be parsed correctly."""
        config = {'lr': 0.001, 'batch': 32}
        exp_id = start_experiment(postgres_connection, 'Test', config=config)
        result = get_experiment(postgres_connection, exp_id)
        assert result['config'] == config

    @pytest.mark.postgres
    def test_metrics_included(self, postgres_connection, seed_test_data):
        """Should include associated metrics."""
        exp_id = start_experiment(postgres_connection, 'Test')
        save_metrics(postgres_connection, exp_id, {'accuracy': 0.95, 'f1': 0.92}, export_to_file=False)

        result = get_experiment(postgres_connection, exp_id)
        assert 'metrics' in result
        assert 'accuracy' in result['metrics']
        assert 'f1' in result['metrics']

    @pytest.mark.postgres
    def test_all_fields_present(self, postgres_connection, seed_test_data):
        """Should return all expected fields."""
        exp_id = start_experiment(postgres_connection, 'Test')
        result = get_experiment(postgres_connection, exp_id)

        expected_fields = ['id', 'name', 'notebook', 'embedding_model', 'config_hash',
                          'config', 'techniques', 'started_at', 'completed_at', 'status',
                          'notes', 'metrics']
        for field in expected_fields:
            assert field in result


class TestCompareExperiments:
    """Tests for compare_experiments() function."""

    @pytest.mark.postgres
    def test_side_by_side_comparison(self, postgres_connection, seed_test_data):
        """Should compare metrics side-by-side."""
        exp1 = start_experiment(postgres_connection, 'Exp1', embedding_model_alias='bge_base_en_v1_5')
        exp2 = start_experiment(postgres_connection, 'Exp2', embedding_model_alias='bge_small_en_v1_5')

        save_metrics(postgres_connection, exp1, {'accuracy': 0.95, 'f1': 0.92}, export_to_file=False)
        save_metrics(postgres_connection, exp2, {'accuracy': 0.88, 'f1': 0.85}, export_to_file=False)

        df = compare_experiments(postgres_connection, [exp1, exp2])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    @pytest.mark.postgres
    def test_metric_selection(self, postgres_connection, seed_test_data):
        """Should allow selecting specific metrics."""
        exp1 = start_experiment(postgres_connection, 'Exp1')
        save_metrics(postgres_connection, exp1,
                    {'accuracy': 0.95, 'precision': 0.92, 'recall': 0.88, 'f1': 0.90},
                    export_to_file=False)

        df = compare_experiments(postgres_connection, [exp1], metric_names=['accuracy', 'f1'])
        assert 'accuracy' in df.columns or 'accuracy' in str(df.index)

    @pytest.mark.postgres
    def test_empty_experiment_list(self, postgres_connection, seed_test_data):
        """Should handle empty experiment list gracefully."""
        df = compare_experiments(postgres_connection, [])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    @pytest.mark.postgres
    def test_single_experiment(self, postgres_connection, seed_test_data):
        """Should handle comparison of single experiment."""
        exp_id = start_experiment(postgres_connection, 'Solo')
        save_metrics(postgres_connection, exp_id, {'accuracy': 0.95}, export_to_file=False)

        df = compare_experiments(postgres_connection, [exp_id])
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 1

    @pytest.mark.postgres
    def test_multiple_experiments_comparison(self, postgres_connection, seed_test_data):
        """Should handle multiple experiments."""
        exp_ids = []
        for i in range(4):
            exp_id = start_experiment(postgres_connection, f'Exp {i}')
            save_metrics(postgres_connection, exp_id, {'accuracy': 0.90 + (i * 0.01)}, export_to_file=False)
            exp_ids.append(exp_id)

        df = compare_experiments(postgres_connection, exp_ids)
        assert len(df) >= 1


# ============================================================================
# Integration Tests
# ============================================================================


class TestFullWorkflow:
    """End-to-end workflow tests combining multiple functions."""

    @pytest.mark.postgres
    def test_complete_experiment_workflow(self, postgres_connection, seed_test_data):
        """Test complete workflow from registration to comparison."""
        # Register embedding
        register_embedding(postgres_connection, 'workflow_model', 'Workflow Model', 768, 100)

        # Start experiments
        exp1 = start_experiment(
            postgres_connection,
            'Baseline',
            embedding_model_alias='workflow_model',
            config={'method': 'baseline'},
            techniques=['basic_search']
        )

        exp2 = start_experiment(
            postgres_connection,
            'Enhanced',
            embedding_model_alias='workflow_model',
            config={'method': 'enhanced'},
            techniques=['reranking', 'query_expansion']
        )

        # Save metrics
        save_metrics(postgres_connection, exp1, {'accuracy': 0.80, 'f1': 0.78}, export_to_file=False)
        save_metrics(postgres_connection, exp2, {'accuracy': 0.92, 'f1': 0.90}, export_to_file=False)

        # Complete experiments
        complete_experiment(postgres_connection, exp1, notes='Baseline run complete')
        complete_experiment(postgres_connection, exp2, notes='Enhanced run complete')

        # Verify
        exp1_data = get_experiment(postgres_connection, exp1)
        exp2_data = get_experiment(postgres_connection, exp2)

        assert exp1_data['status'] == 'completed'
        assert exp2_data['status'] == 'completed'
        assert exp1_data['metrics']['accuracy']['value'] < exp2_data['metrics']['accuracy']['value']

        # Compare
        comparison = compare_experiments(postgres_connection, [exp1, exp2])
        assert len(comparison) == 2

    @pytest.mark.postgres
    def test_embedding_registry_to_experiment_linkage(self, postgres_connection):
        """Test linking experiments to registered embeddings."""
        # Register embedding
        register_embedding(postgres_connection, 'test_embed', 'Test Embedding', 768, 5000,
                          chunk_source_dataset='Wikipedia', chunk_size_config=512,
                          metadata={'version': '1.0'})

        # Get metadata
        metadata = get_embedding_metadata(postgres_connection, 'test_embed')
        assert metadata['dimension'] == 768

        # Use in experiment
        exp_id = start_experiment(
            postgres_connection,
            'Using registered embedding',
            embedding_model_alias='test_embed'
        )

        # Verify linkage
        exp_data = get_experiment(postgres_connection, exp_id)
        assert exp_data['embedding_model'] == 'test_embed'
