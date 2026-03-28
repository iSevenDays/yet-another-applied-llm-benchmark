# Project-Specific Instructions for Yet Another Applied LLM Benchmark

## Environment Setup

### Required Conda Environment
- **IMPORTANT**: Always activate the `yetanother` conda environment before running Python scripts in this project
- Command: `conda activate yetanother`
- This environment contains the necessary dependencies for running the LLM benchmark tests

## Running Tests

### ⚠️ IMPORTANT: Production Test Execution Limitations

**DO NOT run `main.py --run-tests` without timeout for testing purposes!**

```bash
# ❌ NEVER run this without timeout - will take HOURS to complete:
python main.py --model openai_Qwen3-235B-A22B-Instruct-2507 --run-tests --parallel=1

# ✅ Only acceptable for compilation/startup checking (max 10-30s timeout):
conda activate yetanother && timeout 10s python main.py --model openai_Qwen3-235B-A22B-Instruct-2507 --run-tests --parallel=1
```

**Why this limitation exists:**
- `main.py --run-tests` executes ALL ~120 benchmark tests
- Each test requires GPU inference and can take minutes to complete
- Full execution takes **hours** and consumes significant computational resources
- Only use short timeouts to verify compilation/startup, not actual functionality

### For Development Testing

**IMPORTANT: Use appropriate test categories for different development needs**

See [TESTS.md](./TESTS.md) for complete test execution guide.

#### Fast Development Testing (5-10 seconds)
**Use logic-only unit tests for rapid feedback:**
```bash
conda activate yetanother
python -m pytest unittests/test_polyglot_logic_unit.py unittests/test_strip_think.py unittests/test_cache.py -v
```

#### Integration Testing (2+ minutes, requires Docker)
**Use integration tests only when verifying Docker-based execution:**
```bash
conda activate yetanother
python -m pytest unittests/test_integration_polyglot.py -m integration --tb=short
```

#### All Unit Tests (may timeout due to Docker tests)
```bash
conda activate yetanother
timeout 300 python -m pytest unittests/ -v --tb=short -x
```

**Test Categories:**
- **🚀 Fast Tests**: Logic-only, no Docker - use for development
- **🐳 Docker Tests**: Integration tests requiring containers - use sparingly

### Production Test Execution

When running full benchmark evaluation:

```bash
conda activate yetanother
python main.py --model gpt-3.5-turbo --run-tests --generate-report
```

## Visual Test Handling

- VISUAL tests (using `LLMVisionRun`) are automatically detected and **completely skipped** when `vision_eval_llm` is set to `None` in `llm.py`
- Tests that require vision capabilities are not executed at all, preventing unnecessary API calls and execution time
- The system displays "SKIP: [test_name] (requires vision LLM but none configured)" messages for skipped tests
- This ensures efficient benchmark execution and prevents errors when vision model dependencies are not available

## Project Context

This benchmark framework evaluates LLM capabilities on real-world programming tasks using a dataflow DSL. It includes security-first design with Docker/Podman isolation and comprehensive test coverage across multiple programming languages and domains.