# Test Documentation

## Cache System Issue Resolution

### Problem
High cache miss rates (79%) when restarting benchmark runs, despite having cached results available.

### Root Cause
Inconsistent cache key generation patterns created structural mismatches:
- **Legacy cache entries**: 2-component keys `(conversation, hparams)`
- **Current benchmark calls**: 3-component keys `(conversation, max_tokens, hparams)`
- **Vision test calls**: Mixed patterns with explicit `max_tokens=512`

### Solution
Implemented parameter normalization in `llm_cache.py`:
```python
# Before: Inconsistent key structures
key = (conversation, max_tokens, hparams)  # Some calls
key = (conversation, hparams)              # Other calls

# After: Normalized structure
key = (conversation, normalized_hparams)   # All calls
```

### Key Changes
1. **Parameter Normalization**: All `max_tokens` values merged into `hparams` consistently
2. **Backward Compatibility**: Generated keys match legacy 2-component structure
3. **Explicit Precedence**: Direct parameters override `hparams` values
4. **Test Coverage**: Added backward compatibility tests to prevent regression

### Verification
```bash
# All calling patterns now generate identical keys:
cache.get_cache_key(conv, hparams={'max_tokens': 4096})                    # Legacy
cache.get_cache_key(conv, max_tokens=4096, hparams={'max_tokens': 4096})   # Current  
cache.get_cache_key(conv, max_tokens=4096)                                 # Vision

# Result: Same 2-component key structure for all
```

### Impact
- ✅ **Cache hit rate**: Expected to reach ~99% on restarts
- ✅ **Backward compatibility**: Works with existing 281 cache entries
- ✅ **Future-proof**: Prevents similar issues from parameter inconsistencies
- ✅ **Test coverage**: 12/12 unit tests passing including new compatibility tests

### Files Modified
- `llm_cache.py`: Parameter normalization logic
- `unittests/test_cache.py`: Backward compatibility test coverage

### Testing
```bash
conda activate yetanother
python -m pytest unittests/test_cache.py -v
```

## Test Execution Guide

### Test Categories

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

### Execution Recommendations

#### For Development and Quick Validation
**Use fast unit tests** - they verify core logic without Docker overhead:
```bash
conda activate yetanother
python -m pytest unittests/test_polyglot_logic_unit.py unittests/test_strip_think.py -v
```

#### For Full Integration Testing
**Use integration tests** only when you need to verify Docker-based execution:
```bash
conda activate yetanother
python -m pytest unittests/test_integration_polyglot.py -m integration --tb=short
```

#### For Comprehensive Testing (CI/CD)
Run all tests with timeout protection:
```bash
conda activate yetanother
timeout 300 python -m pytest unittests/ -v --tb=short -x
```

### Test Timeout Behavior

- **Fast tests**: Complete in 5-10 seconds
- **Docker tests**: May timeout after 2 minutes due to container startup overhead
- **Recommendation**: Separate fast tests from integration tests in your workflow

### Adding New Tests

**For logic-only tests:** Place in fast unit test files (no Docker dependencies)
**For execution testing:** Use integration test files with proper Docker setup and `@pytest.mark.integration` markers