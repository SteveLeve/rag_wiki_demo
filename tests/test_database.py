"""
Database integration tests.

Tests the PostgreSQL schema, constraints, and data integrity including:
- Table creation and structure
- Index creation and performance
- Foreign key constraints
- CHECK constraints
- JSONB column operations
- Array column operations
- CASCADE delete behavior
- Performance baselines

Coverage target: All schema components

RUNNING THESE TESTS:
===================

1. Ensure PostgreSQL is running with rag_test_db database:
   docker run -d --name pgvector-rag-test \
     -e POSTGRES_PASSWORD=postgres \
     -e POSTGRES_DB=rag_test_db \
     -p 5432:5432 \
     -v pgvector_test_data:/var/lib/postgresql/data \
     pgvector/pgvector:pg16

2. Install test dependencies:
   pip install -e ".[test]"

3. Run all tests:
   pytest tests/test_database.py -v

4. Run specific test class:
   pytest tests/test_database.py::TestSchemaCreation -v

5. Run with markers:
   pytest tests/test_database.py -m "not slow" -v
   pytest tests/test_database.py -m postgres -v

6. Run with coverage:
   pytest tests/test_database.py --cov=tests --cov-report=html

TEST STATISTICS:
================
- 25 test methods
- 8 test classes
- 6 database indexes verified
- 4 tables schema validated
- All CHECK constraints tested
- All FOREIGN KEY constraints tested
- JSONB and array operations covered
- Performance baselines established
"""

import pytest
import psycopg2
from psycopg2 import IntegrityError, DataError
import json
import time
from datetime import datetime


class TestSchemaCreation:
    """Tests for database schema creation."""

    @pytest.mark.integration
    @pytest.mark.postgres
    def test_all_tables_exist(self, postgres_connection):
        """All 4 tables should exist after schema setup."""
        with postgres_connection.cursor() as cur:
            cur.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name
            """)
            tables = [row[0] for row in cur.fetchall()]

        expected_tables = [
            'embedding_registry',
            'evaluation_groundtruth',
            'evaluation_results',
            'experiments'
        ]
        assert set(expected_tables).issubset(set(tables)), \
            f"Expected tables {expected_tables} not found in {tables}"

    @pytest.mark.integration
    @pytest.mark.postgres
    def test_embedding_registry_structure(self, postgres_connection):
        """embedding_registry should have correct columns and types."""
        with postgres_connection.cursor() as cur:
            cur.execute("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'embedding_registry'
                ORDER BY ordinal_position
            """)
            columns = {row[0]: row[1] for row in cur.fetchall()}

        expected_columns = {
            'id': 'integer',
            'model_alias': 'character varying',
            'model_name': 'character varying',
            'dimension': 'integer',
            'embedding_count': 'integer',
            'chunk_source_dataset': 'character varying',
            'chunk_size_config': 'integer',
            'metadata_json': 'jsonb',
            'created_at': 'timestamp without time zone',
            'last_accessed': 'timestamp without time zone',
        }

        for col, dtype in expected_columns.items():
            assert col in columns, f"Column {col} not found"
            assert columns[col] == dtype, \
                f"Column {col} has type {columns[col]}, expected {dtype}"

    @pytest.mark.integration
    @pytest.mark.postgres
    def test_evaluation_groundtruth_structure(self, postgres_connection):
        """evaluation_groundtruth should have correct columns and types."""
        with postgres_connection.cursor() as cur:
            cur.execute("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'evaluation_groundtruth'
                ORDER BY ordinal_position
            """)
            columns = {row[0]: row[1] for row in cur.fetchall()}

        expected_columns = {
            'id': 'integer',
            'question': 'text',
            'source_type': 'character varying',
            'relevant_chunk_ids': 'integer[]',
            'quality_rating': 'character varying',
            'human_notes': 'text',
            'created_at': 'timestamp without time zone',
            'created_by': 'character varying',
        }

        for col, dtype in expected_columns.items():
            assert col in columns, f"Column {col} not found"
            assert columns[col] == dtype, \
                f"Column {col} has type {columns[col]}, expected {dtype}"

    @pytest.mark.integration
    @pytest.mark.postgres
    def test_experiments_structure(self, postgres_connection):
        """experiments should have correct columns and types."""
        with postgres_connection.cursor() as cur:
            cur.execute("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'experiments'
                ORDER BY ordinal_position
            """)
            columns = {row[0]: row[1] for row in cur.fetchall()}

        expected_columns = {
            'id': 'integer',
            'experiment_name': 'character varying',
            'notebook_path': 'character varying',
            'embedding_model_alias': 'character varying',
            'config_hash': 'character varying',
            'config_json': 'jsonb',
            'techniques_applied': 'text[]',
            'started_at': 'timestamp without time zone',
            'completed_at': 'timestamp without time zone',
            'status': 'character varying',
            'notes': 'text',
        }

        for col, dtype in expected_columns.items():
            assert col in columns, f"Column {col} not found"
            assert columns[col] == dtype, \
                f"Column {col} has type {columns[col]}, expected {dtype}"

    @pytest.mark.integration
    @pytest.mark.postgres
    def test_evaluation_results_structure(self, postgres_connection):
        """evaluation_results should have correct columns and types."""
        with postgres_connection.cursor() as cur:
            cur.execute("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'evaluation_results'
                ORDER BY ordinal_position
            """)
            columns = {row[0]: row[1] for row in cur.fetchall()}

        expected_columns = {
            'id': 'integer',
            'experiment_id': 'integer',
            'metric_name': 'character varying',
            'metric_value': 'double precision',
            'metric_details_json': 'jsonb',
            'computed_at': 'timestamp without time zone',
        }

        for col, dtype in expected_columns.items():
            assert col in columns, f"Column {col} not found"
            assert columns[col] == dtype, \
                f"Column {col} has type {columns[col]}, expected {dtype}"

    @pytest.mark.integration
    @pytest.mark.postgres
    def test_all_indexes_exist(self, postgres_connection):
        """All 6 indexes should be created."""
        with postgres_connection.cursor() as cur:
            cur.execute("""
                SELECT indexname FROM pg_indexes
                WHERE schemaname = 'public'
                AND tablename IN ('embedding_registry', 'experiments', 'evaluation_results', 'evaluation_groundtruth')
                ORDER BY indexname
            """)
            indexes = [row[0] for row in cur.fetchall()]

        expected_indexes = [
            'idx_experiments_embedding_model',
            'idx_experiments_started',
            'idx_experiments_status',
            'idx_groundtruth_quality',
            'idx_results_experiment',
            'idx_results_metric',
        ]

        for idx in expected_indexes:
            assert idx in indexes, \
                f"Expected index {idx} not found in {indexes}"

    @pytest.mark.integration
    @pytest.mark.postgres
    def test_index_on_experiments_embedding_model(self, postgres_connection):
        """Index on experiments.embedding_model_alias should exist."""
        with postgres_connection.cursor() as cur:
            cur.execute("""
                SELECT indexdef FROM pg_indexes
                WHERE indexname = 'idx_experiments_embedding_model'
            """)
            result = cur.fetchone()
            assert result is not None, "Index idx_experiments_embedding_model not found"
            assert 'embedding_model_alias' in result[0]

    @pytest.mark.integration
    @pytest.mark.postgres
    def test_index_on_experiments_status(self, postgres_connection):
        """Index on experiments.status should exist."""
        with postgres_connection.cursor() as cur:
            cur.execute("""
                SELECT indexdef FROM pg_indexes
                WHERE indexname = 'idx_experiments_status'
            """)
            result = cur.fetchone()
            assert result is not None, "Index idx_experiments_status not found"
            assert 'status' in result[0]

    @pytest.mark.integration
    @pytest.mark.postgres
    def test_index_on_evaluation_results(self, postgres_connection):
        """Index on evaluation_results.experiment_id should exist."""
        with postgres_connection.cursor() as cur:
            cur.execute("""
                SELECT indexdef FROM pg_indexes
                WHERE indexname = 'idx_results_experiment'
            """)
            result = cur.fetchone()
            assert result is not None, "Index idx_results_experiment not found"


class TestConstraints:
    """Tests for database constraints."""

    @pytest.mark.integration
    @pytest.mark.postgres
    def test_unique_constraint_model_alias(self, postgres_connection):
        """model_alias should have UNIQUE constraint."""
        with postgres_connection.cursor() as cur:
            # Insert first record
            cur.execute("""
                INSERT INTO embedding_registry (model_alias, model_name, dimension)
                VALUES ('test_alias_unique', 'test_model', 768)
            """)
            postgres_connection.commit()

            # Try duplicate - should fail
            with pytest.raises(IntegrityError):
                cur.execute("""
                    INSERT INTO embedding_registry (model_alias, model_name, dimension)
                    VALUES ('test_alias_unique', 'another_model', 384)
                """)
                postgres_connection.commit()

    @pytest.mark.integration
    @pytest.mark.postgres
    def test_check_constraint_source_type_valid(self, postgres_connection):
        """Valid source_type values should be accepted."""
        valid_types = ['llm_generated', 'template_based', 'manual']

        with postgres_connection.cursor() as cur:
            for i, source_type in enumerate(valid_types):
                cur.execute("""
                    INSERT INTO evaluation_groundtruth (question, source_type, relevant_chunk_ids)
                    VALUES (%s, %s, ARRAY[1,2,3])
                """, (f"Question {i}?", source_type))
            postgres_connection.commit()

            # Verify all were inserted
            cur.execute("SELECT COUNT(*) FROM evaluation_groundtruth")
            count = cur.fetchone()[0]
            assert count >= 3

    @pytest.mark.integration
    @pytest.mark.postgres
    def test_check_constraint_source_type_invalid(self, postgres_connection):
        """Invalid source_type should be rejected."""
        with postgres_connection.cursor() as cur:
            with pytest.raises(IntegrityError):
                cur.execute("""
                    INSERT INTO evaluation_groundtruth (question, source_type, relevant_chunk_ids)
                    VALUES ('Bad question?', 'invalid_type', ARRAY[1,2,3])
                """)
                postgres_connection.commit()

    @pytest.mark.integration
    @pytest.mark.postgres
    def test_check_constraint_quality_rating_valid(self, postgres_connection):
        """Valid quality_rating values should be accepted."""
        valid_ratings = ['good', 'bad', 'ambiguous', 'rejected']

        with postgres_connection.cursor() as cur:
            for i, rating in enumerate(valid_ratings):
                cur.execute("""
                    INSERT INTO evaluation_groundtruth (question, source_type, quality_rating)
                    VALUES (%s, 'manual', %s)
                """, (f"Question {i}?", rating))
            postgres_connection.commit()

            # Verify all were inserted
            cur.execute("SELECT COUNT(*) FROM evaluation_groundtruth")
            count = cur.fetchone()[0]
            assert count >= 4

    @pytest.mark.integration
    @pytest.mark.postgres
    def test_check_constraint_quality_rating_invalid(self, postgres_connection):
        """Invalid quality_rating should be rejected."""
        with postgres_connection.cursor() as cur:
            with pytest.raises(IntegrityError):
                cur.execute("""
                    INSERT INTO evaluation_groundtruth (question, source_type, quality_rating)
                    VALUES ('Bad question?', 'manual', 'poor_quality')
                """)
                postgres_connection.commit()

    @pytest.mark.integration
    @pytest.mark.postgres
    def test_check_constraint_status_valid(self, postgres_connection, seed_test_data):
        """Valid status values should be accepted."""
        valid_statuses = ['running', 'completed', 'failed']

        with postgres_connection.cursor() as cur:
            for i, status in enumerate(valid_statuses):
                cur.execute("""
                    INSERT INTO experiments (experiment_name, embedding_model_alias, status)
                    VALUES (%s, 'bge_base_en_v1_5', %s)
                """, (f"Experiment {i}", status))
            postgres_connection.commit()

            # Verify all were inserted
            cur.execute("SELECT COUNT(*) FROM experiments WHERE status IN ('running', 'completed', 'failed')")
            count = cur.fetchone()[0]
            assert count >= 3

    @pytest.mark.integration
    @pytest.mark.postgres
    def test_check_constraint_status_invalid(self, postgres_connection, seed_test_data):
        """Invalid status should be rejected."""
        with postgres_connection.cursor() as cur:
            with pytest.raises(IntegrityError):
                cur.execute("""
                    INSERT INTO experiments (experiment_name, embedding_model_alias, status)
                    VALUES ('Bad experiment', 'bge_base_en_v1_5', 'invalid_status')
                """)
                postgres_connection.commit()


class TestForeignKeys:
    """Tests for foreign key constraints."""

    @pytest.mark.integration
    @pytest.mark.postgres
    def test_fk_experiments_to_registry_valid(self, postgres_connection, seed_test_data):
        """Valid FK reference to embedding_registry should work."""
        with postgres_connection.cursor() as cur:
            # Insert with valid FK
            cur.execute("""
                INSERT INTO experiments (experiment_name, embedding_model_alias)
                VALUES ('test_exp', 'bge_base_en_v1_5')
                RETURNING id
            """)
            postgres_connection.commit()
            exp_id = cur.fetchone()[0]

            # Verify insertion
            cur.execute("SELECT id FROM experiments WHERE id = %s", (exp_id,))
            assert cur.fetchone() is not None

    @pytest.mark.integration
    @pytest.mark.postgres
    def test_fk_experiments_to_registry_invalid(self, postgres_connection, seed_test_data):
        """Invalid FK reference to embedding_registry should fail."""
        with postgres_connection.cursor() as cur:
            with pytest.raises(IntegrityError):
                cur.execute("""
                    INSERT INTO experiments (experiment_name, embedding_model_alias)
                    VALUES ('bad_exp', 'nonexistent_model')
                """)
                postgres_connection.commit()

    @pytest.mark.integration
    @pytest.mark.postgres
    def test_fk_results_to_experiments_valid(self, postgres_connection, seed_test_data):
        """Valid FK from evaluation_results to experiments should work."""
        with postgres_connection.cursor() as cur:
            # Create experiment
            cur.execute("""
                INSERT INTO experiments (experiment_name, embedding_model_alias)
                VALUES ('test_exp', 'bge_base_en_v1_5')
                RETURNING id
            """)
            postgres_connection.commit()
            exp_id = cur.fetchone()[0]

            # Insert result with valid FK
            cur.execute("""
                INSERT INTO evaluation_results (experiment_id, metric_name, metric_value)
                VALUES (%s, 'precision@5', 0.85)
                RETURNING id
            """, (exp_id,))
            postgres_connection.commit()
            result_id = cur.fetchone()[0]

            # Verify insertion
            cur.execute("SELECT id FROM evaluation_results WHERE id = %s", (result_id,))
            assert cur.fetchone() is not None

    @pytest.mark.integration
    @pytest.mark.postgres
    def test_fk_results_to_experiments_invalid(self, postgres_connection, seed_test_data):
        """Invalid FK from evaluation_results to experiments should fail."""
        with postgres_connection.cursor() as cur:
            with pytest.raises(IntegrityError):
                cur.execute("""
                    INSERT INTO evaluation_results (experiment_id, metric_name, metric_value)
                    VALUES (99999, 'precision@5', 0.85)
                """)
                postgres_connection.commit()

    @pytest.mark.integration
    @pytest.mark.postgres
    def test_cascade_delete_results_on_experiment_delete(self, postgres_connection, seed_test_data):
        """Deleting experiment should CASCADE delete results."""
        with postgres_connection.cursor() as cur:
            # Create experiment
            cur.execute("""
                INSERT INTO experiments (experiment_name, embedding_model_alias)
                VALUES ('cascade_test', 'bge_base_en_v1_5')
                RETURNING id
            """)
            postgres_connection.commit()
            exp_id = cur.fetchone()[0]

            # Insert multiple results
            for i in range(3):
                cur.execute("""
                    INSERT INTO evaluation_results (experiment_id, metric_name, metric_value)
                    VALUES (%s, %s, %s)
                """, (exp_id, f'metric_{i}', 0.5 + i * 0.1))
            postgres_connection.commit()

            # Verify results exist
            cur.execute("SELECT COUNT(*) FROM evaluation_results WHERE experiment_id = %s", (exp_id,))
            count_before = cur.fetchone()[0]
            assert count_before == 3

            # Delete experiment
            cur.execute("DELETE FROM experiments WHERE id = %s", (exp_id,))
            postgres_connection.commit()

            # Results should be deleted (CASCADE)
            cur.execute("SELECT COUNT(*) FROM evaluation_results WHERE experiment_id = %s", (exp_id,))
            count_after = cur.fetchone()[0]
            assert count_after == 0


class TestJSONBOperations:
    """Tests for JSONB column operations."""

    @pytest.mark.integration
    @pytest.mark.postgres
    def test_store_and_retrieve_metadata_json(self, postgres_connection):
        """Should store and retrieve complex JSON in metadata_json."""
        with postgres_connection.cursor() as cur:
            metadata = {
                "preserve_existing": True,
                "timestamp": "2024-12-30",
                "nested": {"key": "value", "count": 42}
            }

            cur.execute("""
                INSERT INTO embedding_registry (model_alias, model_name, dimension, metadata_json)
                VALUES (%s, %s, %s, %s)
                RETURNING id
            """, ('test_jsonb', 'test_model', 768, json.dumps(metadata)))
            postgres_connection.commit()
            record_id = cur.fetchone()[0]

            cur.execute("SELECT metadata_json FROM embedding_registry WHERE id = %s", (record_id,))
            retrieved = cur.fetchone()[0]

            assert retrieved == metadata

    @pytest.mark.integration
    @pytest.mark.postgres
    def test_store_config_json_in_experiments(self, postgres_connection, seed_test_data):
        """Should store complex config JSON in experiments."""
        with postgres_connection.cursor() as cur:
            config = {
                "top_n": 5,
                "threshold": 0.75,
                "techniques": ["reranking", "query_expansion"],
                "parameters": {
                    "max_tokens": 512,
                    "temperature": 0.7
                }
            }

            cur.execute("""
                INSERT INTO experiments (experiment_name, embedding_model_alias, config_json)
                VALUES (%s, %s, %s)
                RETURNING id
            """, ('config_test', 'bge_base_en_v1_5', json.dumps(config)))
            postgres_connection.commit()
            exp_id = cur.fetchone()[0]

            cur.execute("SELECT config_json FROM experiments WHERE id = %s", (exp_id,))
            retrieved = cur.fetchone()[0]

            assert retrieved == config

    @pytest.mark.integration
    @pytest.mark.postgres
    def test_store_metric_details_json(self, postgres_connection, seed_test_data):
        """Should store metric details JSON in evaluation_results."""
        with postgres_connection.cursor() as cur:
            # Create experiment first
            cur.execute("""
                INSERT INTO experiments (experiment_name, embedding_model_alias)
                VALUES ('metrics_test', 'bge_base_en_v1_5')
                RETURNING id
            """)
            postgres_connection.commit()
            exp_id = cur.fetchone()[0]

            metric_details = {
                "per_query_precision": [0.8, 0.9, 0.7],
                "mean": 0.8,
                "std_dev": 0.08,
                "breakdown": {
                    "easy": 0.9,
                    "medium": 0.8,
                    "hard": 0.6
                }
            }

            cur.execute("""
                INSERT INTO evaluation_results (experiment_id, metric_name, metric_value, metric_details_json)
                VALUES (%s, 'precision@5', 0.8, %s)
                RETURNING id
            """, (exp_id, json.dumps(metric_details)))
            postgres_connection.commit()
            result_id = cur.fetchone()[0]

            cur.execute("SELECT metric_details_json FROM evaluation_results WHERE id = %s", (result_id,))
            retrieved = cur.fetchone()[0]

            assert retrieved == metric_details

    @pytest.mark.integration
    @pytest.mark.postgres
    def test_query_jsonb_nested_field(self, postgres_connection):
        """Should query nested JSONB fields with JSON operators."""
        with postgres_connection.cursor() as cur:
            # Insert records with different metadata
            metadata1 = {"model_type": "dense", "version": 1}
            metadata2 = {"model_type": "sparse", "version": 2}

            cur.execute("""
                INSERT INTO embedding_registry (model_alias, model_name, dimension, metadata_json)
                VALUES (%s, %s, %s, %s)
            """, ('dense_model', 'dense', 768, json.dumps(metadata1)))

            cur.execute("""
                INSERT INTO embedding_registry (model_alias, model_name, dimension, metadata_json)
                VALUES (%s, %s, %s, %s)
            """, ('sparse_model', 'sparse', 768, json.dumps(metadata2)))
            postgres_connection.commit()

            # Query for specific metadata value
            cur.execute("""
                SELECT model_alias FROM embedding_registry
                WHERE metadata_json->>'model_type' = 'dense'
            """)
            results = [row[0] for row in cur.fetchall()]
            assert 'dense_model' in results
            assert 'sparse_model' not in results

    @pytest.mark.integration
    @pytest.mark.postgres
    def test_update_jsonb_field(self, postgres_connection):
        """Should update nested JSONB fields."""
        with postgres_connection.cursor() as cur:
            metadata = {"count": 100, "updated": False}

            cur.execute("""
                INSERT INTO embedding_registry (model_alias, model_name, dimension, metadata_json)
                VALUES (%s, %s, %s, %s)
                RETURNING id
            """, ('update_test', 'test_model', 768, json.dumps(metadata)))
            postgres_connection.commit()
            record_id = cur.fetchone()[0]

            # Update nested field
            cur.execute("""
                UPDATE embedding_registry
                SET metadata_json = jsonb_set(metadata_json, '{updated}', 'true')
                WHERE id = %s
            """, (record_id,))
            postgres_connection.commit()

            cur.execute("SELECT metadata_json FROM embedding_registry WHERE id = %s", (record_id,))
            updated = cur.fetchone()[0]
            assert updated['updated'] is True


class TestArrayOperations:
    """Tests for array column operations."""

    @pytest.mark.integration
    @pytest.mark.postgres
    def test_store_integer_array_chunk_ids(self, postgres_connection):
        """Should store integer arrays (chunk IDs)."""
        with postgres_connection.cursor() as cur:
            chunk_ids = [1, 2, 3, 4, 5]
            cur.execute("""
                INSERT INTO evaluation_groundtruth (question, source_type, relevant_chunk_ids)
                VALUES ('Test?', 'manual', %s)
                RETURNING relevant_chunk_ids
            """, (chunk_ids,))
            postgres_connection.commit()
            result = cur.fetchone()[0]
            assert result == chunk_ids

    @pytest.mark.integration
    @pytest.mark.postgres
    def test_store_text_array_techniques(self, postgres_connection, seed_test_data):
        """Should store text arrays (techniques)."""
        with postgres_connection.cursor() as cur:
            techniques = ['reranking', 'query_expansion', 'rag_fusion']
            cur.execute("""
                INSERT INTO experiments (experiment_name, embedding_model_alias, techniques_applied)
                VALUES ('tech_test', 'bge_base_en_v1_5', %s)
                RETURNING techniques_applied
            """, (techniques,))
            postgres_connection.commit()
            result = cur.fetchone()[0]
            assert result == techniques

    @pytest.mark.integration
    @pytest.mark.postgres
    def test_query_array_contains(self, postgres_connection, seed_test_data):
        """Should query with array containment operators."""
        with postgres_connection.cursor() as cur:
            # Insert records with different chunk IDs
            cur.execute("""
                INSERT INTO evaluation_groundtruth (question, source_type, relevant_chunk_ids)
                VALUES ('Q1?', 'manual', ARRAY[1,2,3])
            """)
            cur.execute("""
                INSERT INTO evaluation_groundtruth (question, source_type, relevant_chunk_ids)
                VALUES ('Q2?', 'manual', ARRAY[4,5,6])
            """)
            postgres_connection.commit()

            # Query for array containing specific value
            cur.execute("""
                SELECT question FROM evaluation_groundtruth
                WHERE relevant_chunk_ids @> ARRAY[2]
            """)
            results = [row[0] for row in cur.fetchall()]
            assert 'Q1?' in results
            assert 'Q2?' not in results

    @pytest.mark.integration
    @pytest.mark.postgres
    def test_null_vs_empty_array(self, postgres_connection):
        """Should distinguish NULL vs empty array."""
        with postgres_connection.cursor() as cur:
            # Insert with NULL
            cur.execute("""
                INSERT INTO evaluation_groundtruth (question, source_type, relevant_chunk_ids)
                VALUES ('Null array?', 'manual', NULL)
                RETURNING id
            """)
            postgres_connection.commit()
            null_id = cur.fetchone()[0]

            # Insert with empty array
            cur.execute("""
                INSERT INTO evaluation_groundtruth (question, source_type, relevant_chunk_ids)
                VALUES ('Empty array?', 'manual', ARRAY[]::INT[])
                RETURNING id
            """)
            postgres_connection.commit()
            empty_id = cur.fetchone()[0]

            # Verify NULL
            cur.execute("""
                SELECT relevant_chunk_ids FROM evaluation_groundtruth WHERE id = %s
            """, (null_id,))
            null_result = cur.fetchone()[0]
            assert null_result is None

            # Verify empty array
            cur.execute("""
                SELECT relevant_chunk_ids FROM evaluation_groundtruth WHERE id = %s
            """, (empty_id,))
            empty_result = cur.fetchone()[0]
            assert empty_result == []

    @pytest.mark.integration
    @pytest.mark.postgres
    def test_array_length_and_operations(self, postgres_connection):
        """Should support array length and element operations."""
        with postgres_connection.cursor() as cur:
            cur.execute("""
                INSERT INTO evaluation_groundtruth (question, source_type, relevant_chunk_ids)
                VALUES ('Array ops?', 'manual', ARRAY[10,20,30,40,50])
                RETURNING id
            """)
            postgres_connection.commit()
            record_id = cur.fetchone()[0]

            # Check array length
            cur.execute("""
                SELECT array_length(relevant_chunk_ids, 1)
                FROM evaluation_groundtruth WHERE id = %s
            """, (record_id,))
            length = cur.fetchone()[0]
            assert length == 5


class TestDataIntegrity:
    """Tests for data integrity and relationships."""

    @pytest.mark.integration
    @pytest.mark.postgres
    def test_default_timestamps(self, postgres_connection):
        """Should set default timestamps on insert."""
        with postgres_connection.cursor() as cur:
            cur.execute("""
                INSERT INTO embedding_registry (model_alias, model_name, dimension)
                VALUES ('ts_test', 'test', 768)
                RETURNING created_at, last_accessed
            """)
            postgres_connection.commit()
            created_at, last_accessed = cur.fetchone()
            assert created_at is not None
            assert last_accessed is not None
            assert isinstance(created_at, datetime)
            assert isinstance(last_accessed, datetime)

    @pytest.mark.integration
    @pytest.mark.postgres
    def test_default_status_running(self, postgres_connection, seed_test_data):
        """experiments.status should default to 'running'."""
        with postgres_connection.cursor() as cur:
            cur.execute("""
                INSERT INTO experiments (experiment_name, embedding_model_alias)
                VALUES ('default_status_test', 'bge_base_en_v1_5')
                RETURNING status
            """)
            postgres_connection.commit()
            status = cur.fetchone()[0]
            assert status == 'running'

    @pytest.mark.integration
    @pytest.mark.postgres
    def test_default_empty_techniques_array(self, postgres_connection, seed_test_data):
        """techniques_applied should default to empty array."""
        with postgres_connection.cursor() as cur:
            cur.execute("""
                INSERT INTO experiments (experiment_name, embedding_model_alias)
                VALUES ('default_array_test', 'bge_base_en_v1_5')
                RETURNING techniques_applied
            """)
            postgres_connection.commit()
            techniques = cur.fetchone()[0]
            assert techniques == []

    @pytest.mark.integration
    @pytest.mark.postgres
    def test_experiment_workflow(self, postgres_connection, seed_test_data):
        """Should support complete experiment workflow."""
        with postgres_connection.cursor() as cur:
            # Create experiment
            cur.execute("""
                INSERT INTO experiments (
                    experiment_name, embedding_model_alias, status,
                    config_json, techniques_applied
                )
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
            """, (
                'workflow_test',
                'bge_base_en_v1_5',
                'running',
                json.dumps({"top_n": 5}),
                ['reranking']
            ))
            postgres_connection.commit()
            exp_id = cur.fetchone()[0]

            # Add results
            for i in range(3):
                cur.execute("""
                    INSERT INTO evaluation_results
                    (experiment_id, metric_name, metric_value, metric_details_json)
                    VALUES (%s, %s, %s, %s)
                """, (
                    exp_id,
                    f'metric_{i}',
                    0.5 + i * 0.1,
                    json.dumps({"query_count": i + 1})
                ))
            postgres_connection.commit()

            # Mark as completed
            cur.execute("""
                UPDATE experiments
                SET status = 'completed', completed_at = CURRENT_TIMESTAMP
                WHERE id = %s
            """, (exp_id,))
            postgres_connection.commit()

            # Verify complete state
            cur.execute("""
                SELECT status, completed_at FROM experiments WHERE id = %s
            """, (exp_id,))
            status, completed_at = cur.fetchone()
            assert status == 'completed'
            assert completed_at is not None

            # Verify results are linked
            cur.execute("""
                SELECT COUNT(*) FROM evaluation_results WHERE experiment_id = %s
            """, (exp_id,))
            count = cur.fetchone()[0]
            assert count == 3


class TestPerformance:
    """Performance baseline tests."""

    @pytest.mark.integration
    @pytest.mark.postgres
    @pytest.mark.slow
    def test_insert_1000_embeddings_under_5s(self, postgres_connection):
        """Inserting 1000 embeddings should take < 5 seconds."""
        import time

        start_time = time.time()

        with postgres_connection.cursor() as cur:
            # Batch insert
            for i in range(1000):
                cur.execute("""
                    INSERT INTO embedding_registry (model_alias, model_name, dimension, embedding_count)
                    VALUES (%s, %s, %s, %s)
                """, (f'perf_model_{i}', f'Model {i}', 768, i))
            postgres_connection.commit()

        elapsed_time = time.time() - start_time

        # Should complete in reasonable time (5 seconds)
        assert elapsed_time < 5.0, \
            f"Inserting 1000 embeddings took {elapsed_time:.2f}s, expected < 5.0s"

    @pytest.mark.integration
    @pytest.mark.postgres
    def test_query_with_filter_under_100ms(self, postgres_connection, seed_test_data):
        """Filtered queries should execute quickly."""
        with postgres_connection.cursor() as cur:
            start_time = time.time()

            cur.execute("""
                SELECT id, experiment_name FROM experiments
                WHERE status = 'completed'
            """)
            results = cur.fetchall()

            elapsed_time = time.time() - start_time

            # Should complete under 100ms
            assert elapsed_time < 0.1, \
                f"Query took {elapsed_time*1000:.2f}ms, expected < 100ms"

    @pytest.mark.integration
    @pytest.mark.postgres
    def test_join_query_performance(self, postgres_connection, seed_test_data):
        """Join queries should be efficient with foreign keys."""
        with postgres_connection.cursor() as cur:
            # Create test data
            cur.execute("""
                INSERT INTO experiments (experiment_name, embedding_model_alias)
                VALUES ('perf_test', 'bge_base_en_v1_5')
                RETURNING id
            """)
            postgres_connection.commit()
            exp_id = cur.fetchone()[0]

            # Insert results
            for i in range(100):
                cur.execute("""
                    INSERT INTO evaluation_results
                    (experiment_id, metric_name, metric_value)
                    VALUES (%s, %s, %s)
                """, (exp_id, f'metric_{i}', 0.5))
            postgres_connection.commit()

            # Time join query
            start_time = time.time()

            cur.execute("""
                SELECT e.experiment_name, r.metric_name, r.metric_value
                FROM experiments e
                JOIN evaluation_results r ON e.id = r.experiment_id
                WHERE e.id = %s
            """, (exp_id,))
            results = cur.fetchall()

            elapsed_time = time.time() - start_time

            assert len(results) == 100
            assert elapsed_time < 0.5, \
                f"Join query took {elapsed_time*1000:.2f}ms, expected < 500ms"

    @pytest.mark.integration
    @pytest.mark.postgres
    def test_index_utilization_status_filter(self, postgres_connection, seed_test_data):
        """Status index should make filtering efficient."""
        with postgres_connection.cursor() as cur:
            # Create many experiments with different statuses
            for i in range(100):
                status = ['running', 'completed', 'failed'][i % 3]
                cur.execute("""
                    INSERT INTO experiments (experiment_name, embedding_model_alias, status)
                    VALUES (%s, %s, %s)
                """, (f'index_test_{i}', 'bge_base_en_v1_5', status))
            postgres_connection.commit()

            # Query with index
            start_time = time.time()

            cur.execute("""
                SELECT COUNT(*) FROM experiments WHERE status = 'completed'
            """)
            count = cur.fetchone()[0]

            elapsed_time = time.time() - start_time

            assert count > 0
            assert elapsed_time < 0.1, \
                f"Indexed query took {elapsed_time*1000:.2f}ms, expected < 100ms"


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    @pytest.mark.integration
    @pytest.mark.postgres
    def test_very_long_text_fields(self, postgres_connection):
        """Should handle very long text in text fields."""
        long_text = "x" * 50000
        with postgres_connection.cursor() as cur:
            cur.execute("""
                INSERT INTO evaluation_groundtruth (question, source_type, human_notes)
                VALUES (%s, 'manual', %s)
                RETURNING id
            """, (long_text, long_text))
            postgres_connection.commit()
            record_id = cur.fetchone()[0]

            cur.execute("""
                SELECT question FROM evaluation_groundtruth WHERE id = %s
            """, (record_id,))
            retrieved = cur.fetchone()[0]
            assert retrieved == long_text

    @pytest.mark.integration
    @pytest.mark.postgres
    def test_special_characters_in_text(self, postgres_connection):
        """Should handle special characters and escaping."""
        special_text = "Question with 'quotes', \"double quotes\", and $pecial ch@rs"
        with postgres_connection.cursor() as cur:
            cur.execute("""
                INSERT INTO evaluation_groundtruth (question, source_type)
                VALUES (%s, 'manual')
                RETURNING question
            """, (special_text,))
            postgres_connection.commit()
            retrieved = cur.fetchone()[0]
            assert retrieved == special_text

    @pytest.mark.integration
    @pytest.mark.postgres
    def test_large_integer_arrays(self, postgres_connection):
        """Should handle large integer arrays."""
        large_array = list(range(1000))
        with postgres_connection.cursor() as cur:
            cur.execute("""
                INSERT INTO evaluation_groundtruth (question, source_type, relevant_chunk_ids)
                VALUES ('Array test?', 'manual', %s)
                RETURNING relevant_chunk_ids
            """, (large_array,))
            postgres_connection.commit()
            retrieved = cur.fetchone()[0]
            assert retrieved == large_array

    @pytest.mark.integration
    @pytest.mark.postgres
    def test_deeply_nested_jsonb(self, postgres_connection):
        """Should handle deeply nested JSON structures."""
        deep_json = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "value": "deep"
                        }
                    }
                }
            }
        }
        with postgres_connection.cursor() as cur:
            cur.execute("""
                INSERT INTO embedding_registry (model_alias, model_name, dimension, metadata_json)
                VALUES (%s, %s, %s, %s)
                RETURNING metadata_json
            """, ('deep_json', 'test', 768, json.dumps(deep_json)))
            postgres_connection.commit()
            retrieved = cur.fetchone()[0]
            assert retrieved == deep_json

    @pytest.mark.integration
    @pytest.mark.postgres
    def test_float_precision_in_metrics(self, postgres_connection, seed_test_data):
        """Should preserve float precision in metric values."""
        with postgres_connection.cursor() as cur:
            # Create experiment
            cur.execute("""
                INSERT INTO experiments (experiment_name, embedding_model_alias)
                VALUES ('float_test', 'bge_base_en_v1_5')
                RETURNING id
            """)
            postgres_connection.commit()
            exp_id = cur.fetchone()[0]

            # Insert with high precision float
            precise_value = 0.123456789
            cur.execute("""
                INSERT INTO evaluation_results (experiment_id, metric_name, metric_value)
                VALUES (%s, 'test', %s)
                RETURNING metric_value
            """, (exp_id, precise_value))
            postgres_connection.commit()
            retrieved = cur.fetchone()[0]

            # Check precision is maintained (double precision = ~15 significant digits)
            assert abs(retrieved - precise_value) < 1e-10
