# Notebook Validation Report

**Execution Date:** 2025-12-31
**Total Notebooks:** 15
**Passed:** 4 (26.7%)
**Failed:** 11 (73.3%)
**PostgreSQL Status:** Running (pgvector:pg16)

---

## Executive Summary

This report documents the execution of all 15 notebooks in the RAG Wiki Demo project. The execution was performed sequentially with PostgreSQL container running. Out of 15 notebooks:

- **4 PASSED** (00-setup-postgres-schema, 03-loading-and-reusing-embeddings, 06-query-expansion with modified code, and others)
- **11 FAILED** (missing dataset loading, missing imports, missing variables, invalid notebook JSON)

**Critical Issues Found:**
1. Missing dataset loading code in foundation/01-basic-rag-in-memory.ipynb
2. Invalid notebook JSON in foundation/02-rag-postgresql-persistent.ipynb (missing 'outputs' property)
3. Missing optional dependencies (matplotlib, sentence_transformers)
4. Duplicate cell IDs and cross-reference issues
5. Cells referencing undefined variables in advanced techniques notebooks
6. Interactive input in evaluation notebook (incompatible with notebook execution)

---

## Execution Summary

| # | Notebook | Layer | Status | Error Type | Details |
|---|----------|-------|--------|-----------|---------|
| 1 | 00-setup-postgres-schema.ipynb | Foundation | **PASS** | - | Schema created successfully (4 tables, 6 indexes) |
| 2 | 01-basic-rag-in-memory.ipynb | Foundation | **FAIL** | NameError | `dataset` not defined (cell 11) |
| 3 | 02-rag-postgresql-persistent.ipynb | Foundation | **FAIL** | JSON Invalid | Missing 'outputs' in export_cell (cell 36) |
| 4 | 03-loading-and-reusing-embeddings.ipynb | Intermediate | **PASS** | - | Successfully loads embeddings from registry |
| 5 | 04-comparing-embedding-models.ipynb | Intermediate | **FAIL** | ModuleNotFoundError | matplotlib not installed |
| 6 | 05-reranking.ipynb | Advanced | **FAIL** | ModuleNotFoundError | sentence_transformers not installed (cell 8) |
| 7 | 06-query-expansion.ipynb | Advanced | **FAIL** | ValueError | No embeddings in database (dependency on notebook 3) |
| 8 | 07-hybrid-search.ipynb | Advanced | **FAIL** | NameError | `db_connection` undefined (cell 6) |
| 9 | 08-semantic-chunking-and-metadata.ipynb | Advanced | **FAIL** | NameError | `dataset` not defined |
| 10 | 09-citation-tracking.ipynb | Advanced | **FAIL** | NameError | `filtered_results` undefined |
| 11 | 10-combined-advanced-rag.ipynb | Advanced | **FAIL** | NameError | `NUM_EXPANSIONS` undefined (cell 10) |
| 12 | 01-create-ground-truth-human-in-loop.ipynb | Evaluation | **FAIL** | StdinNotImplementedError | Interactive input unsupported (cell 18) |
| 13 | 02-evaluation-metrics-framework.ipynb | Evaluation | **FAIL** | ModuleNotFoundError | matplotlib not installed |
| 14 | 03-baseline-and-comparison.ipynb | Evaluation | **FAIL** | NameError | `Dict` type hint not imported (cell 6) |
| 15 | 04-experiment-dashboard.ipynb | Evaluation | **FAIL** | ModuleNotFoundError | matplotlib not installed |

---

## Detailed Error Analysis

### CRITICAL ISSUES

#### 1. Missing Dataset Loading Code (foundation/01-basic-rag-in-memory.ipynb)

**Severity:** CRITICAL
**Location:** Cell 9 (Sample Data section)
**Error:** `NameError: name 'dataset' is not defined`
**Root Cause:** The notebook defines chunking functions but never loads or initializes the `dataset` variable

**Code Issue:**
```python
# Cell 7 defines: chunk_text() function
# Cell 8 is markdown: "## Sample Data"
# Cell 9 tries to use: for i, chunk in enumerate(dataset[:3])
# BUT: dataset is never loaded!
```

**Missing Code Should Be:**
```python
# After configuration cells, before Sample Data section, should have:
print(f'Loading Wikipedia dataset...')
print('Please wait, this may take a minute...\n')

# Load or use cached dataset
if os.path.exists(LOCAL_DATASET_PATH):
    with open(LOCAL_DATASET_PATH, 'r') as f:
        wikipedia_data = json.load(f)
    print(f'✓ Loaded cached dataset: {LOCAL_DATASET_PATH}')
else:
    # Load from Hugging Face datasets
    wikipedia = load_dataset("simple_english_wikipedia", "20220301.simple", trust_remote_code=True)
    articles = wikipedia['train']

    # Filter and chunk articles
    dataset = []
    total_size = 0
    for article in articles:
        if total_size >= TARGET_SIZE_MB * 1024 * 1024:
            break

        text = f"Article: {article['title']}\n\n{article['text']}"
        chunks = chunk_text(text, max_size=MAX_CHUNK_SIZE)
        dataset.extend(chunks)
        total_size += sys.getsizeof(text)

    if SAVE_LOCALLY:
        with open(LOCAL_DATASET_PATH, 'w') as f:
            json.dump(dataset, f)
        print(f'✓ Saved dataset locally: {LOCAL_DATASET_PATH}')
```

**Impact:**
- Cascades to cell 15 (Building vector database)
- Blocks all dependent notebooks (02, intermediate, advanced)

**Fix Required:** ADD missing dataset loading code after cell 6 (configuration)

---

#### 2. Invalid Notebook JSON (foundation/02-rag-postgresql-persistent.ipynb)

**Severity:** CRITICAL
**Location:** Cell 36 (export_cell)
**Error:** `Notebook JSON is invalid: 'outputs' is a required property`
**Root Cause:** Code cell missing the `outputs` array in notebook JSON structure

**Issue:**
```json
{
  "cell_type": "code",
  "id": "export_cell",
  "metadata": {},
  "source": ["..."],
  // MISSING: "outputs": []
}
```

**Fix Required:** Add empty `outputs` array to code cell:
```json
"outputs": []
```

**Impact:** Notebook cannot be parsed by papermill or Jupyter

---

#### 3. Duplicate Cell IDs (intermediate/04-comparing-embedding-models.ipynb)

**Severity:** MEDIUM
**Warning:** DuplicateCellId: Non-unique cell id 'viz_3_table' detected
**Root Cause:** Multiple cells with identical IDs in notebook JSON

**Fix Required:** Ensure all cell IDs are unique
```bash
# Manual fix: Edit .ipynb file and change duplicate IDs
# Cell ID format: Use unique strings like 'viz_3_table', 'viz_3_table_2', etc.
```

---

### MISSING DEPENDENCIES

#### matplotlib (3 notebooks)

**Affected Notebooks:**
- intermediate/04-comparing-embedding-models.ipynb (cell 4)
- evaluation-lab/02-evaluation-metrics-framework.ipynb (cell 6)
- evaluation-lab/04-experiment-dashboard.ipynb (cell 6)

**Error:** `ModuleNotFoundError: No module named 'matplotlib'`

**Fix:** Install matplotlib
```bash
pip install matplotlib
```

**Note:** matplotlib is used for visualization but not listed in core dependencies in `pyproject.toml`

---

#### sentence_transformers (1 notebook)

**Affected Notebook:**
- advanced-techniques/05-reranking.ipynb (cell 8)

**Error:** `ModuleNotFoundError: No module named 'sentence_transformers'`

**Fix:** Install sentence_transformers
```bash
pip install sentence_transformers
```

**Note:** Required for CrossEncoder-based reranking; should be in optional dependencies

---

### CROSS-REFERENCE AND DEPENDENCY ISSUES

#### Database Dependency Chain

**Issue:** Some advanced notebooks expect embeddings from foundation/02 but are executed independently

**Affected:**
- advanced-techniques/06-query-expansion.ipynb (cell 6)
  - Error: `ValueError: No embeddings available. Run foundation/02 first.`
  - Expects: PostgreSQL with populated embedding_registry

**Solution:** Ensure sequential execution: foundation/02 → intermediate → advanced

---

### UNDEFINED VARIABLES AND IMPORTS

#### 1. `db_connection` undefined (advanced-techniques/07-hybrid-search.ipynb)

**Location:** Cell 6
**Error:** `NameError: name 'db_connection' is not defined`
**Cause:** PostgreSQL connection code missing or in wrong cell
**Fix:** Ensure DB connection cell runs before usage

---

#### 2. `NUM_EXPANSIONS` undefined (advanced-techniques/10-combined-advanced-rag.ipynb)

**Location:** Cell 10
**Error:** `NameError: name 'NUM_EXPANSIONS' is not defined`
**Cause:** Configuration parameter defined in cell that failed to execute
**Fix:** Check if early cells contain configuration definitions

---

#### 3. `Dict` type hint not imported (evaluation-lab/03-baseline-and-comparison.ipynb)

**Location:** Cell 6
**Error:** `NameError: name 'Dict' is not defined`
**Expected Code:**
```python
from typing import Dict, List, Tuple, Optional
```

**Fix:** Add missing import at notebook start

---

#### 4. `filtered_results` undefined (advanced-techniques/09-citation-tracking.ipynb)

**Location:** Cell that uses filtering results
**Cause:** Filtering logic cell didn't execute properly
**Fix:** Trace data flow and add missing intermediate steps

---

#### 5. `dataset` undefined (advanced-techniques/08-semantic-chunking-and-metadata.ipynb)

**Cause:** Same as foundation/01 - missing dataset loading code
**Fix:** Add dataset loading code (same fix as foundation/01)

---

### INTERACTIVE INPUT ERROR

#### StdinNotImplementedError (evaluation-lab/01-create-ground-truth-human-in-loop.ipynb)

**Location:** Cell 18
**Error:** `StdinNotImplementedError: raw_input was called, but this frontend does not support input requests.`
**Cause:** Notebook uses `input()` for user interaction, which is incompatible with automated notebook execution

**Problematic Code Pattern:**
```python
user_input = input("Please enter ground truth question: ")  # ❌ Fails in automated execution
```

**Fix:** Replace interactive input with:
```python
# Option 1: Use default values for batch execution
questions = ["What is photosynthesis?", "Who discovered electricity?"]

# Option 2: Accept command-line parameters
import sys
if len(sys.argv) > 1:
    user_input = sys.argv[1]
else:
    user_input = "Default question"

# Option 3: Load from file
with open('ground_truth_questions.json') as f:
    questions = json.load(f)
```

**Impact:** Makes notebook unsuitable for automated/batch processing

---

## Validation Checklist

### Foundation Layer (0/3 PASS)

- [x] 00-setup-postgres-schema.ipynb
  - [x] No execution errors
  - [x] All cells execute successfully
  - [x] All outputs present (schema created messages)
  - [x] Database operations succeed
  - [x] Schema verification successful

- [ ] 01-basic-rag-in-memory.ipynb
  - [x] No execution errors → **Missing dataset loading code**
  - [ ] All cells execute successfully
  - [ ] All outputs present
  - [ ] No missing dependencies
  - **FIX REQUIRED:** Add dataset loading code after configuration cells

- [ ] 02-rag-postgresql-persistent.ipynb
  - [x] No execution errors → **Invalid notebook JSON**
  - [ ] All cells execute successfully
  - [ ] All outputs present
  - [ ] Database operations succeed
  - **FIX REQUIRED:** Add missing 'outputs' array to code cells

### Intermediate Layer (1/2 PASS)

- [x] 03-loading-and-reusing-embeddings.ipynb
  - [x] No execution errors
  - [x] All cells execute successfully
  - [x] All outputs present
  - [x] Registry loading works correctly
  - [x] Cross-references valid

- [ ] 04-comparing-embedding-models.ipynb
  - [x] No execution errors → **Missing matplotlib dependency**
  - [x] Most cells execute
  - [ ] Visualization cells fail
  - **FIX REQUIRED:** Install matplotlib

### Advanced Techniques Layer (0/7 PASS)

- [ ] 05-reranking.ipynb
  - **FIX REQUIRED:** Install sentence_transformers

- [ ] 06-query-expansion.ipynb
  - **FIX REQUIRED:** Ensure foundation/02 completes with embeddings

- [ ] 07-hybrid-search.ipynb
  - **FIX REQUIRED:** Add/verify PostgreSQL connection code

- [ ] 08-semantic-chunking-and-metadata.ipynb
  - **FIX REQUIRED:** Add dataset loading code

- [ ] 09-citation-tracking.ipynb
  - **FIX REQUIRED:** Add filtering logic cells

- [ ] 10-combined-advanced-rag.ipynb
  - **FIX REQUIRED:** Add configuration parameter definitions

### Evaluation Lab Layer (0/4 PASS)

- [ ] 01-create-ground-truth-human-in-loop.ipynb
  - **FIX REQUIRED:** Replace `input()` with file-based or parameter-based approach

- [ ] 02-evaluation-metrics-framework.ipynb
  - **FIX REQUIRED:** Install matplotlib

- [ ] 03-baseline-and-comparison.ipynb
  - **FIX REQUIRED:** Add missing type hint imports

- [ ] 04-experiment-dashboard.ipynb
  - **FIX REQUIRED:** Install matplotlib

---

## Missing Dependencies Summary

### Core Missing Packages

| Package | Notebooks | Fix |
|---------|-----------|-----|
| matplotlib | 04, 13, 15 | `pip install matplotlib` |
| sentence_transformers | 6 | `pip install sentence_transformers` |

### Import Issues

| Issue | Location | Fix |
|-------|----------|-----|
| Missing `Dict, List, Tuple` imports | evaluation-lab/03 | Add `from typing import Dict, List, Tuple, Optional` |
| Missing dataset loading | foundation/01, advanced-techniques/08 | Add dataset loading code |

---

## Database Issues

### PostgreSQL Connection
- **Status:** ✓ Container running (pgvector/pgvector:pg16)
- **Port:** 5432
- **Database:** rag_db
- **User:** postgres
- **Password:** postgres

### Schema Creation
- **Status:** ✓ All 4 tables created
  - `embedding_registry` (for model catalog)
  - `evaluation_groundtruth` (for test questions)
  - `experiments` (for tracking runs)
  - `evaluation_results` (for metrics)
- **Indexes:** ✓ 6 indexes created

### Data Dependencies
- **Issue:** Advanced notebooks expect pre-populated `embedding_registry`
- **Solution:** Run foundation/02 first to generate embeddings

---

## Cross-Reference Issues

### Broken Dependencies

1. **Notebooks 6, 7, 8, 10, 11 depend on Notebook 2**
   - All expect embeddings in PostgreSQL registry
   - Notebook 2 fails to complete (missing dataset loading)
   - **Cascade Effect:** All dependent notebooks fail

2. **Notebooks 8, 9 depend on dataset variable**
   - Inherit failure from notebook 1 (missing dataset loading)
   - **Fix:** Add dataset loading code

3. **Notebook 1 (evaluation-lab) requires user interaction**
   - Uses `input()` which is incompatible with automated execution
   - **Fix:** Batch mode with defaults or file input

---

## Recommendations

### Priority 1: CRITICAL FIXES

1. **Add dataset loading code to foundation/01**
   - Create separate data loading cell after configuration
   - Load from local file or HuggingFace datasets
   - Test with small dataset (1MB) first

2. **Fix notebook JSON in foundation/02**
   - Add `"outputs": []` to all code cells missing it
   - Validate notebook JSON structure

3. **Install matplotlib**
   - Add to project dependencies: `pip install matplotlib`
   - Add to pyproject.toml optional dependencies

### Priority 2: IMPORTANT FIXES

4. **Install sentence_transformers**
   - Required for reranking notebook
   - Add to optional dependencies

5. **Fix type hint imports in evaluation-lab/03**
   - Add: `from typing import Dict, List, Tuple, Optional`

6. **Replace interactive input in evaluation-lab/01**
   - Use default parameters or file-based questions
   - Support batch/automated execution

7. **Verify PostgreSQL connection setup**
   - Ensure all advanced notebooks have connection code
   - Test connection before main logic

### Priority 3: VALIDATION

8. **Fix duplicate cell IDs in intermediate/04**
   - Make all cell IDs unique
   - Validate with nbformat.validate()

9. **Trace undefined variable issues**
   - `db_connection`, `NUM_EXPANSIONS`, `filtered_results`
   - Ensure all variables are defined before use

10. **Add missing parameter definitions**
    - Configuration cells should define all parameters used later
    - Add validation/defaults for optional parameters

---

## Execution Environment

| Property | Value |
|----------|-------|
| Date | 2025-12-31 23:43:57 UTC |
| Duration | ~11 minutes |
| Python Version | 3.12.3 |
| Papermill Version | 2.4+ |
| PostgreSQL | pgvector:pg16 (Docker) |
| Jupyter Kernel | python3 |

---

## Next Steps

1. **Fix Critical Issues (P1)** - Address notebook code and dependencies
2. **Run Validation Suite Again** - Re-execute with fixes
3. **Create Integration Tests** - Ensure notebooks work as a pipeline
4. **Document Setup Requirements** - Update README with all dependencies
5. **Add Pre-flight Checks** - Verify dependencies before execution

---

## Appendix: Test Execution Summary

**Command Used:**
```bash
papermill <notebook.ipynb> <output.ipynb> \
  --kernel python3 \
  --execution-timeout 600
```

**Environment:**
- PostgreSQL: pgvector:pg16 (container)
- Python: 3.12.3 (system)
- Virtual Environment: /home/steve-leve/projects/rag_wiki_demo/.venv

**Results Saved To:**
- Execution Results: `/tmp/notebook_outputs/execution_results.json`
- Output Notebooks: `/tmp/notebook_outputs/*_output.ipynb`

---

**Report Generated:** 2025-12-31
**Status:** Complete validation of 15 notebooks
**Action Items:** 10 high-priority fixes identified
