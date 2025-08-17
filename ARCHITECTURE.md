# Yet Another Applied LLM Benchmark - Architecture Documentation

## Overview

This is a comprehensive LLM evaluation framework designed to test language models on real-world programming tasks. The system uses a dataflow Domain Specific Language (DSL) to create sophisticated test pipelines that can evaluate models on diverse coding challenges including code generation, debugging, optimization, and cross-language translation.

## Core Philosophy

- **Real-world Tasks**: All test cases derive from actual tasks the author has asked LLMs to perform
- **Security-First**: All LLM-generated code executes in isolated Docker/Podman containers
- **Composable Testing**: Dataflow DSL allows complex multi-stage evaluation pipelines
- **Automated Evaluation**: Combines code execution with LLM-based judgment for comprehensive assessment

## Project Structure

```
/
├── main.py                     # CLI entry point and test orchestration
├── evaluator.py               # Core DSL framework and node implementations
├── llm.py                     # LLM abstraction layer and model management
├── docker_controller.py       # Container isolation and execution management
├── create_results_html.py     # HTML report generation
├── config.json.example        # Configuration template
├── Dockerfile                 # Multi-language container environment
├── requirements.txt           # Python dependencies
├── requirements-extra.txt     # Extended model support dependencies
├── tests/                     # Test case implementations (~100 tests)
│   ├── print_hello.py         # Simple test example
│   ├── convert_to_c.py        # Complex code translation test
│   └── ...
├── llms/                      # Model adapter implementations
│   ├── openai_model.py        # OpenAI API integration
│   ├── anthropic_model.py     # Anthropic API integration
│   ├── ollama_model.py        # Local model support
│   └── ...
├── evaluation_examples/       # Generated HTML reports and assets
└── results/                   # Cached test results by git commit
    └── [commit_hash]/
        └── [model]-run[N].p   # Pickled test results
```

## Core Architecture Components

### 1. Dataflow DSL (evaluator.py)

The heart of the system is a computation graph framework where test cases are built by chaining Node objects using operators:

```python
# Example test case structure:
'Write a "hello world" program in python' >> LLMRun() >> ExtractCode() >> PythonRun() >> SubstringEvaluator("hello world")
```

**Key DSL Operators:**
- `>>` (ThenNode): Sequential execution - output of left node feeds into right node
- `&` (AndNode): Logical AND - both nodes must succeed
- `|` (OrNode): Logical OR - either node can succeed  
- `~` (NotNode): Logical negation

**Core Node Types:**

**LLM Interaction Nodes:**
- `LLMRun()`: Query primary model under test
- `LLMConversation()`: Multi-turn conversation with state
- `LLMVisionRun()`: Vision model evaluation for image outputs

**Code Execution Nodes:**
- `PythonRun()`: Execute Python code in container
- `CRun()`: Compile and run C code
- `RustRun()`: Compile and run Rust code
- `BashRun()`: Execute bash scripts
- `SQLRun()`: Execute SQL queries

**Processing Nodes:**
- `ExtractCode()`: Extract code blocks from LLM responses
- `ExtractJSON()`: Parse JSON from responses
- `MakeFile()`: Create files in container environment

**Evaluation Nodes:**
- `SubstringEvaluator()`: Check if output contains specific text
- `RegexEvaluator()`: Pattern matching evaluation
- `EqualEvaluator()`: Exact match evaluation
- `JSONSubsetEvaluator()`: Structured data validation

**Control Flow Nodes:**
- `UntilDone()`: Loop until condition met (max 100 iterations)
- `Setup()`: Initialize container environment
- `Echo()`: Debug node for pipeline inspection

### 2. LLM Abstraction Layer (llm.py)

Provides unified interface across different model providers with caching, retry logic, and streaming support.

**Key Features:**
- **Universal API**: Consistent interface across OpenAI, Anthropic, Mistral, Gemini, Ollama, etc.
- **Smart Caching**: Pickle-based response caching with cache key generation
- **Retry Logic**: Exponential backoff (10s, 20s, 30s, 60s, 90s, 120s, 300s)
- **Streaming Support**: Real-time response processing with timeout handling
- **Configuration**: JSON-based model and hyperparameter configuration

**Model Support:**
- OpenAI: GPT-3.5, GPT-4, GPT-4o, o1-mini/preview
- Anthropic: Claude 3 (Opus, Sonnet, Haiku), Claude 3.5 Sonnet
- Google: Gemini Pro, Gemini Flash via VertexAI
- Mistral: All Mistral models
- Groq: Fast inference models
- Ollama: Local model deployment
- Cohere: Command models

### 3. Container Isolation (docker_controller.py)

Secure execution environment for untrusted LLM-generated code.

**Security Features:**
- **Complete Isolation**: Docker/Podman containers with no host access
- **Automatic Cleanup**: Containers destroyed after each test
- **Timeout Protection**: 20-second execution limits
- **File Sandboxing**: TAR-based file transfer to containers

**Container Environment:**
- Ubuntu 22.04 base image
- Python 3.12 with scientific computing stack (NumPy, JAX, PyTorch)
- Multi-language support: C/C++, Rust, Swift, SQLite
- Development tools: GCC, Clang, GDB

**Execution Modes:**
- **Safe Mode** (default): Full container isolation
- **Unsafe Mode**: Direct host execution (strongly discouraged)

### 4. Test Orchestration (main.py)

Command-line interface for running benchmarks with support for:

**Execution Modes:**
- Sequential execution (default)
- Parallel execution with configurable worker count
- Individual test selection
- Changed-tests-only mode (git-based)

**Result Management:**
- Git commit-based result versioning
- Pickle serialization for caching
- Multi-run aggregation and comparison
- HTML report generation

**Progress Tracking:**
- Real-time progress bars with pass/fail statistics
- Comprehensive logging with debug output
- Error handling and recovery

### 5. Report Generation (create_results_html.py)

Sophisticated HTML reporting system with:

**Features:**
- **Interactive Grid**: Sortable model vs test comparison
- **Detailed Drill-down**: Per-test execution traces
- **Visual Feedback**: Color-coded success rates
- **Tag Filtering**: Filter tests by category (code, python, c, etc.)
- **Image Support**: Automatic image rendering for visual tests
- **Syntax Highlighting**: Code blocks with language detection

**Report Structure:**
- Main grid: `evaluation_examples/index.html`
- Individual test pages: `evaluation_examples/[test_name].html`
- Detailed execution traces: `evaluation_examples/[test]_[model].html`

## Data Flow Architecture

### Test Execution Pipeline

1. **Test Discovery**: Scan `tests/` directory for Python files with Test* classes
2. **Model Initialization**: Create LLM instances with configuration
3. **Container Setup**: Initialize Docker/Podman environment
4. **Pipeline Execution**: Execute dataflow graph with error handling
5. **Result Collection**: Aggregate success/failure with detailed traces
6. **Report Generation**: Create interactive HTML reports
7. **Cleanup**: Destroy containers and cache results

### Node Execution Model

Each node in the computation graph implements:

```python
def __call__(self, input_data):
    """
    Process input and yield (output, Reason) tuples.
    Supports multiple return values for branching logic.
    """
    yield processed_output, Reason(node_type, execution_details)
```

**Reason Tracking**: Complete execution trace for debugging and reporting

### Configuration System

**config.json Structure:**
```json
{
    "container": "podman|docker",
    "hparams": {
        "temperature": 0.7,
        "max_tokens": 2048
    },
    "llms": {
        "openai": {"api_key": "..."},
        "anthropic": {"api_key": "..."},
        "ollama": {"base_url": "http://localhost:11434"}
    }
}
```

## Test Case Architecture

### Test Case Structure

Every test follows this pattern:

```python
from evaluator import *

DESCRIPTION = "Human-readable test description"
TAGS = ['category1', 'category2']  # For filtering

question = "The prompt to send to the LLM"
TestClassName = question >> Node1() >> Node2() >> ... >> EvaluatorNode()

if __name__ == "__main__":
    print(run_test(TestClassName))
```

### Categories of Tests

**Code Generation** (~30 tests):
- `print_hello.py`: Basic Python output
- `convert_to_c.py`: Cross-language translation  
- `draw_flag_bmp.py`: Visual output generation

**Debugging** (~15 tests):
- `debug_broken_code_parcount.py`: Fix buggy implementations
- `fix_threading_issue.py`: Concurrency problems

**Code Understanding** (~20 tests):
- `explain_code_prime.py`: Algorithm explanation
- `decompile_py_simple.py`: Bytecode reverse engineering

**System Tasks** (~25 tests):
- `bash_renamer.py`: Shell scripting
- `git_merge.py`: Version control operations
- `docker_cuda.py`: DevOps configurations

**Data Processing** (~10 tests):
- `data_table_processing.py`: CSV/table manipulation
- `extract_emails.py`: Text processing with regex

## Security Considerations

### Container Security
- **No Network Access**: Containers run in isolated networks
- **No Host Mounts**: No access to host filesystem
- **Resource Limits**: CPU and memory constraints
- **Ephemeral Containers**: Destroyed after each test

### Code Execution Safety
- **Timeout Protection**: Hard 20-second execution limits
- **Signal Handling**: Proper cleanup on interruption
- **Error Isolation**: Exceptions don't crash the test runner

### Data Safety
- **No Sensitive Data**: Tests use synthetic/public data only
- **No Credential Exposure**: API keys managed securely
- **Audit Trail**: Complete execution logging

## Performance Characteristics

### Execution Speed
- **Container Overhead**: ~2-3 seconds per test for setup/teardown
- **LLM Latency**: Variable (1-30 seconds depending on model)
- **Parallel Scaling**: Near-linear with worker count for I/O-bound tests

### Resource Requirements
- **Memory**: ~2GB base + 500MB per parallel worker
- **Storage**: ~1GB for container images, ~100MB per test run
- **Network**: Model-dependent (100KB-10MB per test)

### Caching Strategy
- **LLM Response Caching**: Persistent pickle cache with prompt hashing
- **Container Reuse**: Limited (security vs performance tradeoff)
- **Result Versioning**: Git commit-based cache invalidation

## Extension Points

### Adding New Models
1. Create adapter in `llms/[provider]_model.py`
2. Implement `make_request()` method
3. Add provider detection logic in `llm.py`
4. Update configuration template

### Adding New Languages
1. Install tools in `Dockerfile`
2. Create `[Language]Run` node in `evaluator.py`
3. Add syntax highlighting support in `create_results_html.py`

### Adding New Test Types
1. Create node class inheriting from `Node`
2. Implement `__call__()` method
3. Add formatting support for reports
4. Write test cases using the new node

## Future Architecture Considerations

### Scalability Improvements
- **Distributed Execution**: Multi-machine test distribution
- **Container Pooling**: Reusable warm containers
- **Result Streaming**: Real-time result updates

### Enhanced Security
- **SELinux/AppArmor**: Additional container hardening
- **Network Policies**: Fine-grained network restrictions
- **Runtime Monitoring**: Container behavior analysis

### Advanced Evaluation
- **Multi-modal Testing**: Code + documentation + images
- **Interactive Testing**: Multi-turn debugging scenarios
- **Performance Benchmarking**: Execution time/memory measurement

This architecture enables systematic evaluation of LLM capabilities on realistic programming tasks while maintaining security, reproducibility, and extensibility.