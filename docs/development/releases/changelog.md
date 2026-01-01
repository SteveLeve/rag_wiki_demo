# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-01-01

### Added

#### Testing Infrastructure (Phase 1)
- 186 comprehensive tests across 5 test files (5,057 lines of test code)
- 9 shared pytest fixtures for database, mocking, test data
- GitHub Actions CI/CD pipeline with matrix testing (Python 3.10, 3.11, 3.12)
- Test markers for categorized execution (@pytest.mark.unit, @pytest.mark.integration, etc.)
- Integration tests with PostgreSQL 16 and pgvector
- End-to-end pipeline tests
- Notebook execution tests using Papermill
- Test coverage tracking and reporting
- Transaction-isolated database testing with automatic rollback

#### Educational Documentation (Phase 3)
- CONCEPTS.md (1,993 lines) - Foundational RAG concepts and techniques
- ADVANCED_CONCEPTS.md (1,432 lines) - Advanced RAG techniques and strategies
- EVALUATION_CONCEPTS.md (1,211 lines) - Evaluation methodologies and metrics
- Total educational content: 4,636+ lines
- 8 Mermaid diagrams for visual learning across documentation
- 50+ runnable code examples
- 132+ inline comments in foundation notebooks explaining functions and algorithms
- Comprehensive parameter and return value documentation

#### Notebook Bug Fixes & Validation (Phase 4)
- Interactive input replaced with batch mode (evaluation-lab/01)
- Missing typing imports added (evaluation-lab/03)
- PostgreSQL connection setup added (advanced-techniques/07)
- Configuration parameter definitions added (advanced-techniques/10)
- All 15 notebooks now execute successfully
- Notebook execution validation reports

#### Dependencies & Configuration
- Optional dependencies group `[viz]` for visualization (matplotlib)
- Optional dependencies group `[advanced]` for advanced techniques (sentence-transformers)
- Consolidated optional dependencies in pyproject.toml
- Updated project version to 2.0.0 in pyproject.toml

#### Documentation Improvements
- TESTING_SUMMARY.md - Comprehensive testing strategy and statistics
- NOTEBOOK_VALIDATION_REPORT.md - Detailed validation results
- NOTEBOOK_FIXES_REPORT.md - Bug fixes applied and verification
- VERIFICATION_REPORT.md - Cloud reference cleanup verification
- CROSS_REFERENCE_REPORT.md - Internal link validation

### Changed

#### Cloud Provider References (Phase 2)
- Removed "Neon (Vercel PostgreSQL)" references from documentation (now "Neon PostgreSQL")
- Removed Vercel deployment documentation
- Removed Cloudflare Workers references
- Updated export functions documentation for vendor neutrality
- Established local-first development paradigm

#### Export Functions
- Generic JSON export format (vendor-neutral)
- pgvector export format (PostgreSQL-specific)
- Pinecone export format (vector database alternative)
- No hardcoded vendor URLs or service-specific code

#### Documentation Structure
- Foundation notebooks include comprehensive inline documentation
- Added function docstrings with parameter and return value descriptions
- Enhanced algorithm step-by-step comments
- Improved cross-reference consistency across all notebooks
- Updated README.md with clear provider options (local, Neon, Supabase)
- QUICK_REFERENCE.md features PostgreSQL providers prominently

#### Database Configuration
- PostgreSQL support highlighted as primary choice
- pgvector as primary vector search extension
- Support documented for: local installations, Neon, Supabase, RDS, self-hosted
- Generic connection patterns enable any PostgreSQL provider

#### Project Metadata
- Updated pyproject.toml version to 2.0.0
- Enhanced keywords in project metadata
- Improved project description
- Updated documentation URLs

### Fixed

#### Critical Notebook Bugs

**foundation/01-basic-rag-in-memory.ipynb**
- Added missing dataset loading code
- Ensures `dataset` variable properly initialized before usage
- Supports both local file caching and HuggingFace datasets loading

**foundation/02-rag-postgresql-persistent.ipynb**
- Fixed invalid notebook JSON (missing `outputs` arrays in code cells)
- Validates notebook structure with nbformat

**intermediate/04-comparing-embedding-models.ipynb**
- Fixed duplicate cell IDs (cell id 'viz_3_table')
- Ensured all cell IDs unique and valid

**advanced-techniques/05-reranking.ipynb**
- Resolved missing sentence_transformers dependency
- Added to optional dependencies [advanced]

**advanced-techniques/07-hybrid-search.ipynb**
- Fixed undefined `db_connection` variable
- Added PostgreSQL connection setup check and initialization
- Connection established before first usage

**advanced-techniques/10-combined-advanced-rag.ipynb**
- Added missing configuration parameter definitions
- NUM_EXPANSIONS, TOP_K_INITIAL, TOP_K_FINAL defined in configuration section
- ENABLE_RERANKING, ENABLE_CITATION_TRACKING set with defaults

#### Dependency Issues
- matplotlib added to optional dependencies (visualization in 3 notebooks)
- sentence-transformers added to optional dependencies (reranking)
- Both available via `pip install -e ".[viz,advanced]"`

#### Documentation Consistency
- Cloud provider references consistent across all layers
- External links verified as functional
- Internal cross-references validated
- Notebook docstrings updated for accuracy

### Removed

#### Vendor Lock-In References
- Removed "Vercel" product references (2 instances in notebook docstrings)
- Removed Cloudflare Workers documentation
- Removed Vectorize product references
- Removed hardcoded Vercel-specific configuration examples

#### Obsolete Content
- Removed vendor-specific deployment patterns
- Removed cloud-provider-exclusive features
- Removed service-specific authentication patterns

### Security

#### No Security Issues
- No vulnerable dependencies identified
- Tests include error handling and constraint validation
- Database transactions properly isolated
- No hardcoded credentials in code

#### Best Practices
- Environment variables for sensitive configuration
- Secure password handling via .env files
- Transaction rollback for data integrity
- Input validation in all utility functions

## [1.0.0] - 2024-XX-XX

### Added
- Initial release with foundational RAG implementation
- 12 working Jupyter notebooks across 4 learning layers
- PostgreSQL + pgvector persistence backend
- Registry system for embedding model tracking
- Experiment tracking and management system
- 7 advanced RAG techniques
  - Reranking with cross-encoders
  - Query expansion and reformulation
  - Hybrid search combining multiple approaches
  - Semantic chunking with metadata
  - Citation tracking for source attribution
  - Combined advanced techniques
- Evaluation framework with metrics computation
- Database schema with 4 tables and 6 indexes
- Support for multiple embedding models (BGE, Sentence Transformers)

### Changed
- Documentation structure evolved to support learning progression
- Schema design refined for performance

### Fixed
- Initial implementation bugs and edge cases

---

## Version Numbering

This project follows [Semantic Versioning](https://semver.org/):

- **MAJOR:** Breaking changes to API or functionality
- **MINOR:** New features maintaining backward compatibility
- **PATCH:** Bug fixes and hotfixes

Examples:
- v2.0.0 - Major release with new phases (testing, cleanup, education, fixes)
- v2.1.0 - Minor release with new embedding models or evaluation metrics
- v2.0.1 - Patch release with bug fix or documentation update

---

## Upgrade Guide

### From v1.0.0 to v2.0.0

**No breaking changes** - All code from v1.0.0 remains compatible.

#### Recommended Updates

1. **Install new dependencies:**
   ```bash
   pip install -e ".[viz,advanced]"
   ```

2. **Review new documentation:**
   - [CONCEPTS.md](CONCEPTS.md) - Foundational understanding
   - [ADVANCED_CONCEPTS.md](ADVANCED_CONCEPTS.md) - Advanced techniques
   - [EVALUATION_CONCEPTS.md](EVALUATION_CONCEPTS.md) - Evaluation methodology

3. **Run updated tests:**
   ```bash
   pytest tests/ -v
   ```

4. **Execute all notebooks:**
   - All 15 notebooks now fully functional
   - No code changes needed on your end
   - Improvements are backward compatible

---

## Known Limitations

### v2.0.0

1. **Ollama Requirement**
   - Text generation requires Ollama server running locally
   - Not suitable for cloud-only deployments
   - Workaround: Deploy Ollama separately

2. **Database Dependency Chain**
   - Advanced notebooks depend on embeddings from foundation/02
   - Execute in recommended order for best results
   - Can work independently with pre-loaded embeddings

3. **Single-Model Focus**
   - Default embedding model: BGE Base (768-dim)
   - Support for alternatives documented
   - Cross-model comparison available in intermediate/04

---

## Deprecation Notices

### v2.0.0

None. All v1.0.0 functionality maintained and enhanced.

### Planned Deprecations

- v2.1.0: Consider deprecating basic Ollama integration in favor of LLM APIs
- v3.0.0: Potential migration from pandas-based experiment tracking to Arrow-based

---

## Contributing

When contributing changes, please:

1. Follow [Keep a Changelog](https://keepachangelog.com/) format
2. Use appropriate section headers (Added, Changed, Fixed, Removed, etc.)
3. Include version and date for new entries
4. Reference related issues or PRs
5. Update version in pyproject.toml if needed
6. Run full test suite: `pytest tests/ -v`

---

## Release Timeline

- **v1.0.0:** 2024 - Initial release with core RAG implementation
- **v2.0.0:** 2026-01-01 - Testing, documentation, cleanup, bug fixes
- **v2.1.0:** TBD - Additional models and advanced metrics
- **v3.0.0:** TBD - Streaming and production deployment patterns

---

For more details about specific releases, see:
- [RELEASE_NOTES.md](RELEASE_NOTES.md) - Detailed v2.0.0 release information
- [TESTING_SUMMARY.md](TESTING_SUMMARY.md) - Testing infrastructure details
- [NOTEBOOK_VALIDATION_REPORT.md](NOTEBOOK_VALIDATION_REPORT.md) - Validation results
- [VERIFICATION_REPORT.md](VERIFICATION_REPORT.md) - Cloud reference cleanup

---

*Changelog last updated: 2026-01-01*
*Format: Keep a Changelog v1.0.0*
*Versioning: Semantic Versioning 2.0.0*
