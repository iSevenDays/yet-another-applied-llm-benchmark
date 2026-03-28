# Test Execution Guide

## Test Categories

This project has two main categories of tests with very different execution characteristics:

#### 🚀 Fast Unit Tests (Recommended for Development)
These tests run in **5-10 seconds** and don't require Docker containers:

**Logic-Only Tests:**
- `unittests/test_polyglot_logic_unit.py` - Polyglot evaluation logic (no Docker)
- `unittests/test_strip_think.py` - Thinking token removal functionality
- `unittests/test_error_reporting.py` - Error message formatting
- `unittests/test_assembly_debug.py` - Assembly emulator logic
- `unittests/test_cache.py` - Cache system functionality

**Command to run fast tests:**
```bash
conda activate yetanother
python -m pytest unittests/test_polyglot_logic_unit.py unittests/test_strip_think.py unittests/test_error_reporting.py unittests/test_assembly_debug.py unittests/test_cache.py -v
```

#### 🐳 Docker-Based Integration Tests (Slow)
These tests require Docker containers and can take **2+ minutes each**:

**Integration Tests:**
- `unittests/test_integration_polyglot.py` - Full polyglot pipeline with Docker
- `unittests/test_rust_run_fix.py` - RustRun Docker execution testing
- `unittests/test_improved_polyglot_evaluator.py` - Complete polyglot evaluation pipeline
- `unittests/test_docker_job.py` - Docker container management
- `unittests/test_sqlite_integration.py` - SQLite Docker integration
- `unittests/test_printhellopoly2_pipeline.py` - Pipeline integration testing

**Command to run integration tests (requires Docker setup):**
```bash
conda activate yetanother
python -m pytest unittests/test_integration_polyglot.py -m integration --tb=short
```

## Execution Recommendations

| Scenario | Command | Time |
|---|---|---|
| Development / quick validation | `python -m pytest unittests/test_polyglot_logic_unit.py unittests/test_strip_think.py unittests/test_error_reporting.py unittests/test_assembly_debug.py unittests/test_cache.py -v` | 5-10s |
| Docker integration testing | `python -m pytest unittests/test_integration_polyglot.py -m integration --tb=short` | 2+ min |
| Comprehensive (CI/CD) | `timeout 300 python -m pytest unittests/ -v --tb=short -x` | up to 5 min |

> **Note:** Always run `conda activate yetanother` first. Separate fast tests from Docker tests in your workflow to avoid unnecessary delays.

### Adding New Tests

**For logic-only tests:** Place in fast unit test files (no Docker dependencies)
**For execution testing:** Use integration test files with proper Docker setup and `@pytest.mark.integration` markers