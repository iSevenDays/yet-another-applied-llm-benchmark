# Yet Another Applied LLM Benchmark - Architecture Documentation

## Overview

This is a comprehensive LLM evaluation framework designed to test language models on real-world programming tasks derived from actual use cases. The system employs a sophisticated dataflow Domain Specific Language (DSL) to create multi-stage test pipelines that can evaluate models on diverse coding challenges including code generation, debugging, optimization, cross-language translation, and system administration tasks.

Unlike academic benchmarks that focus on abstract capabilities, this framework evaluates practical programming assistance - the kind of tasks developers actually ask LLMs to perform in their daily work.

## Core Philosophy

- **Real-world Relevance**: Every test case derives from actual tasks the author has asked LLMs to perform
- **Security-First Design**: All LLM-generated code executes in isolated Docker/Podman containers
- **Composable Testing**: Dataflow DSL enables complex multi-stage evaluation pipelines
- **Automated Evaluation**: Combines code execution with LLM-based judgment for comprehensive assessment
- **Reproducible Science**: Git-based versioning ensures consistent evaluation across code changes
- **Performance at Scale**: Multi-layer caching and parallel execution for efficiency

## Project Structure

```
/
├── main.py                     # CLI entry point and test orchestration
├── evaluator.py               # Core DSL framework and node implementations
├── llm.py                     # LLM abstraction layer and model management
├── llm_cache.py               # Dedicated LLM response caching with backward compatibility
├── docker_controller.py       # Container isolation and execution management
├── create_results_html.py     # HTML report generation
├── config.json.example        # Configuration template
├── Dockerfile                 # Multi-language container environment
├── requirements.txt           # Python dependencies
├── requirements-extra.txt     # Extended model support dependencies
├── tests/                     # Test case implementations (~100 tests)
├── unittests/                 # Unit tests for framework components
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
- `CRun()`: Compile and run C code with GCC
- `CppRun()`: Compile and run C++ code with G++
- `RustRun()`: Compile and run Rust code with cargo
- `SwiftRun()`: Execute Swift code
- `BashRun()`: Execute bash scripts
- `SQLRun()`: Execute SQL queries with SQLite
- `TerminalRun()`: Interactive terminal session

**Processing Nodes:**
- `ExtractCode()`: Extract code blocks from LLM responses with language detection
- `ExtractJSON()`: Parse JSON from responses with validation
- `MakeFile()`: Create files in container environment
- `Setup()`: Initialize container environment with custom files

**Evaluation Nodes:**
- `SubstringEvaluator()`: Check if output contains specific text
- `RegexEvaluator()`: Pattern matching evaluation with full regex support
- `EqualEvaluator()`: Exact match evaluation with type conversion
- `ContainsIntEvaluator()`: Verify integer presence in output
- `PyEvaluator()`: Custom Python evaluation functions
- `JSONSubsetEvaluator()`: Structured data validation for complex JSON

**Control Flow Nodes:**
- `UntilDone()`: Loop until condition met (max 100 iterations)
- `Echo()`: Debug node for pipeline inspection with pretty printing
- `StringNode()`: Constant string values in pipeline

**Logical Operation Nodes:**
- `ThenNode()`: Sequential execution (>>)
- `AndNode()`: Logical AND (&) - both conditions must succeed
- `OrNode()`: Logical OR (|) - either condition can succeed
- `NotNode()`: Logical negation (~)

### 2. LLM Abstraction Layer (llm.py)

Provides unified interface across different model providers with caching, retry logic, and streaming support.

**Key Features:**
- **Universal API**: Consistent interface across OpenAI, Anthropic, Mistral, Gemini, Ollama, etc.
- **Smart Caching**: Dedicated LLMCache class with backward-compatible key generation
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

### 3. Container Isolation & Security (`docker_controller.py`)

Secure execution environment for untrusted LLM-generated code with robust defense-in-depth architecture.

**Security Architecture:**
- **Complete Process Isolation**: Docker/Podman containers with no host filesystem access
- **Network Isolation**: Containers run without network access by default
- **Automatic Cleanup**: Containers destroyed after each test with daemon threads
- **Timeout Protection**: Hard 20-second execution limits with signal handling
- **File Sandboxing**: TAR-based file transfer prevents directory traversal attacks
- **Resource Limits**: CPU and memory constraints prevent resource exhaustion

**Critical Security Flag (`docker_controller.py:40`)**:
```python
I_HAVE_BLIND_FAITH_IN_LLMS_AND_AM_OKAY_WITH_THEM_BRICKING_MY_MACHINE_OR_MAKING_THEM_HALT_AND_CATCH_FIRE = False
```
**WARNING**: Never set to `True` - enables direct host execution bypassing all security

**Container Environment (`Dockerfile`)**:
- **Base**: Ubuntu 22.04 LTS for stability and security updates
- **Python**: 3.12 with scientific computing stack (NumPy, SciPy, JAX, PyTorch)
- **Multi-language Support**: 
  - C/C++: GCC, Clang, GDB for debugging
  - Rust: Full Rust toolchain with Cargo
  - Swift: Swift runtime and compiler
  - Database: SQLite3 for SQL tests
  - Tools: Git, curl, wget, pkg-config
- **Package Management**: Pip with explicit package installation for reproducibility

**Dual Backend Support:**
- **Docker Backend**: Standard Docker daemon integration with full Docker API
- **Podman Backend**: Rootless container execution with subprocess integration for enhanced security

**Container Lifecycle Management with Pooling System:**

The framework implements a sophisticated container pooling system that eliminates setup/teardown overhead while maintaining security isolation:

**Pool Architecture:**
- **Thread-Safe Queue**: Manages pool of warm containers with configurable size limits
- **Container Lifecycle**: Get from pool → Reset state → Execute test → Return to pool
- **Overflow Handling**: Creates new containers when pool is empty, destroys excess when full
- **State Management**: Tracks active containers and ensures proper cleanup on exit

**Pool Operations:**
- **Container Acquisition**: Attempts pool reuse first, creates new container if unavailable
- **State Reset**: Cleans container filesystem between uses while preserving warm state
- **Return Logic**: Returns clean containers to pool, destroys containers that fail reset
- **Graceful Fallback**: Falls back to traditional create/destroy for non-pooled containers

**Pool Benefits:**
- **Performance**: Eliminates 2-3 seconds of setup/teardown per test
- **Resource Efficiency**: Maintains warm containers ready for immediate use
- **Security Maintained**: Container state is reset between uses
- **Automatic Cleanup**: All containers destroyed on process exit

**Interactive Container Sessions (`DockerJob` class)**:
- **Persistent Bash Sessions**: Long-running interactive shells for multi-command tests
- **ANSI Escape Sequence Filtering**: Clean output processing for reliable parsing
- **Robust Error Handling**: Broken pipe detection and graceful degradation
- **Configurable Timeouts**: Multiple timeout strategies for different test types

### 4. Test Orchestration (`main.py`)

Sophisticated test execution engine with parallel processing, real-time monitoring, and comprehensive error handling.

**Execution Modes:**
- **Sequential Mode** (default): Preserves original behavior, optimal for debugging
- **Parallel Mode**: Multiprocessing-based execution with configurable worker count
- **Individual Test Selection**: Run specific tests by name with `--test` flag
- **Delta Mode**: Git-based changed-tests-only execution with `--only-changed`
- **Multi-run Support**: Statistical analysis with `--times` parameter

**Advanced Parallel Architecture:**
- **Multiprocessing Pool**: Distributes test execution across configurable worker processes
- **Asynchronous Execution**: Non-blocking test submission with real-time result collection
- **Progress Monitoring**: Live updates as tests complete rather than batch processing
- **Resource Management**: Worker pool size configurable based on system capabilities

**Key Features**:
- **Process Isolation**: Each test runs in separate Python process for reliability
- **Enhanced Cache Management**: LLM instances now share cache across worker processes with file locking
- **Container Pooling**: Workers reuse warm containers instead of creating new ones
- **Real-time Feedback**: Results processed as they complete, not batched
- **Failure Isolation**: Test failures don't affect other running tests
- **Resource Management**: Configurable worker count based on system resources

**Performance Optimizations**:
- **Shared Caching**: Workers benefit from each other's cached responses (previously disabled)
- **Container Reuse**: Eliminates setup/teardown overhead through pooling
- **File-Level Synchronization**: Thread-safe cache operations across processes

**Result Management System:**
- **Git Commit-based Versioning**: Results organized by commit hash for reproducibility
- **Pickle Serialization**: Efficient binary storage with compression
- **Multi-run Aggregation**: Statistical analysis across multiple test runs
- **Cross-commit Comparison**: Load and compare results across different code versions
- **Atomic Operations**: Safe concurrent access to result files

**Progress Monitoring & Logging (`main.py:448-463`)**:
```python
# Sophisticated logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s - [%(funcName)s] %(message)s',
    handlers=[
        logging.FileHandler('debug_stream.log', mode='a'),  # Persistent debugging
        logging.StreamHandler(sys.stdout)  # Real-time console output
    ]
)
```

**Features**:
- **Real-time Progress Bars**: TQDM-based with pass/fail statistics and ETA
- **Comprehensive Logging**: Debug-level file logging with function-level tracing
- **Error Recovery**: Graceful handling of import errors, timeout exceptions, and Docker failures
- **Resource Monitoring**: Memory usage tracking and worker process management

**Error Handling Hierarchy:**
1. **Import Error Handling**: Graceful skip of malformed test files
2. **Docker Exception Handling**: Specific handling for container failures
3. **Timeout Management**: Signal-based timeout with cleanup
4. **Process Communication**: Broken pipe and subprocess failure recovery
5. **Exception Isolation**: Individual test failures don't crash the runner

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

### Configuration System (`config.json`)

Centralized configuration management with provider-specific settings and global hyperparameters.

**Configuration Structure:**
```json
{
    "container": "podman|docker",
    "hparams": {
        "temperature": 0.7,
        "max_tokens": 2048
    },
    "llms": {
        "openai": {
            "api_key": "sk-...",
            "api_base": "https://api.openai.com/v1",
            "hparams": {"temperature": 0.0}  // Provider-specific overrides
        },
        "anthropic": {
            "api_key": "sk-ant-...",
            "hparams": {"max_tokens": 4096}
        },
        "vertexai": {
            "project_id": "my-gcp-project"
        },
        "ollama": {
            "base_url": "http://localhost:11434",
            "system_prompt": "You are a helpful assistant."
        },
        "groq": {"api_key": "gsk_..."},
        "mistral": {"api_key": "..."},
        "cohere": {"api_key": "..."},
        "moonshot": {"api_key": "..."}
    }
}
```

**Configuration Hierarchy:**
1. **Global hyperparameters**: Applied to all models by default
2. **Provider-specific hyperparameters**: Override global settings per provider
3. **Environment variable overrides**: `OPENAI_BASE_URL`, `OLLAMA_SYSTEM_PROMPT`
4. **Runtime overrides**: Command-line parameter modifications

**Provider-Specific Features:**
- **OpenAI**: Support for multiple API bases, o1-model special handling
- **Anthropic**: Native Claude API integration with message format
- **VertexAI**: Google Cloud integration with project-based authentication
- **Ollama**: Local model deployment with custom system prompts
- **Groq**: High-speed inference optimization
- **Mistral**: Native Mistral API with streaming support

### Model Provider Architecture (`llms/`)

**Universal Model Interface:**
All model providers implement a standardized `make_request()` method:
```python
def make_request(self, conversation, add_image=None, max_tokens=None, json=False, stream=False):
    # Provider-specific implementation
    return response_text
```

**Provider Implementations:**

**OpenAI Model (`openai_model.py`):**
- **Vision Support**: Automatic image encoding for GPT-4V
- **Streaming**: Real-time response processing
- **Model-specific Handling**: Special logic for o1 models (no temperature)
- **JSON Mode**: Structured output generation
- **Base URL Override**: Support for OpenAI-compatible APIs

**Anthropic Model (`anthropic_model.py`):**
- **Message Format**: Native Claude conversation structure
- **Simple Integration**: Direct API mapping without complex preprocessing

**Local Models:**
- **Ollama**: Local deployment with HTTP API
- **LlamaCpp**: Direct Python bindings for local inference

**Cloud Providers:**
- **VertexAI**: Google Cloud integration with project authentication
- **Groq**: Optimized for speed with specialized hardware

**Provider Selection Logic (`llm.py:35-85`):**
```python
def __init__(self, name="gpt-3.5-turbo", use_cache=True):
    if 'openai_' in name:
        self.model = OpenAIModel(name.replace('openai_', ''))
    elif 'gpt' in name or name.startswith('o1'):
        self.model = OpenAIModel(name)
    elif name.startswith('ollama_'):
        self.model = OllamaModel(name.replace('ollama_', ''))
    elif 'claude' in name:
        self.model = AnthropicModel(name)
    # ... additional provider detection
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

### Comprehensive Test Taxonomy (100+ Tests)

**Code Generation & Translation** (~35 tests):
- **Basic Programming**: `print_hello.py`, `print_hello_swift.py`, `program_sqrt.py`
- **Cross-language Translation**: `convert_to_c.py`, `convert_to_c_simple.py`, `call_rust_from_python.py`
- **Visual Programming**: `draw_flag_bmp.py`, `flexbox_webpage.py`, `webgl_triangle.py`
- **Algorithm Implementation**: `quicksort_swift.py`, `implement_crc32.py`, `unholy_matrix.py`
- **Advanced Features**: `python_parallel_wordcount.py`, `rust_parallel_wordcount.py`

**Debugging & Code Repair** (~20 tests):
- **Bug Fixing**: `debug_broken_code_parcount.py`, `fix_threading_issue.py`, `fix_torch_backward.py`
- **Performance Issues**: `debug_innerhtml_eventlistener.py`, `fix_append_vs_extend.py`
- **Syntax Problems**: `fix_json.py`, `fix_node_error.py`, `why_broken_flask_extra_brace.py`
- **Logical Errors**: `fix_with_patch.py`, `fix_tokenizer.py`

**Code Understanding & Analysis** (~25 tests):
- **Algorithm Explanation**: `explain_code_prime.py`, `explain_code_prime2.py`, `explain_vbroadcast.py`
- **Reverse Engineering**: `decompile_py_simple.py`, `decompile_py_mid.py`, `decompile_py_rref.py`
- **Code Comprehension**: `basic_code_understanding.py`, `c_weird_expression.py`
- **Domain Knowledge**: `what_is_automodel.py`, `what_is_inv.py`, `which_package_sbox.py`

**System Administration & DevOps** (~25 tests):
- **Shell Scripting**: `bash_renamer.py`, `bash_find_dont_contain.py`, `bash_convert_not_overwrite.py`
- **Version Control**: `git_merge.py`, `git_cherrypick.py`, `basic_git_setup.py`
- **Containerization**: `docker_cuda.py`, `save_expired_html.py`
- **File Operations**: `change_filetype.py`, `gitignore_anywhere.py`
- **System Configuration**: `emacs_lisp_silence_cmd.py`, `upython_mqtt.py`

**Data Processing & Analysis** (~15 tests):
- **Table Manipulation**: `data_table_processing.py`, `data_extraction_byyear.py`, `data_train_timetable.py`
- **Text Processing**: `extract_emails.py`, `extract_references.py`, `regex_remove_5_words.py`
- **Format Conversion**: `do_uudecode.py`, `identify_uuencode.py`, `make_json.py`
- **Database Operations**: `explore_sql_db.py`, `fancy_sql_process.py`, `make_sqlite_table.py`

**Scientific Computing** (~10 tests):
- **Numerical Computing**: `numpy_advanced_index.py`, `numpy_ix.py`, `faster_l2_diff.py`
- **Machine Learning**: `jax_onehot.py`, `torch_to_jnp.py`, `simulate_torch_grad.py`
- **Optimization**: `numba_levenshtein.py`, `numba_rref.py`, `vectorize_small_update.py`
- **Mathematical Operations**: `c_rref.py`, `strided_trick.py`, `unit_conversion_math.py`

**Domain-Specific Knowledge** (~15 tests):
- **Hardware/Electronics**: `db9_pinout.py`, `rewrite_mac_crypto.py`
- **Academic/Research**: `find_bug_in_paper.py`, `hallucinate_reference.py`, `latex_protect.py`
- **Industry-Specific**: `aws_ipv6.py`, `freecad_construction.py`, `tokenizer_vocab.py`
- **Entertainment/Puzzles**: `emoji_movies.py`, `play_20_questions.py`, `gol_rle_decode.py`

**Special Categories**:
- **Parser Development**: `easy_parser_generator.py`, `program_in_new_assembly.py`
- **Performance Optimization**: `convert_dp_to_iterative.py`, `shorten_c_function.py`
- **Interactive Programming**: `python_chess_game_prefix.py`, `baking_help.py`
- **Security/Encoding**: `base64_qanda.py`, `merge_into_16.py`

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

### Execution Speed (Post-Optimization)
- **Container Overhead**: ~0.1-0.2 seconds per test (with pooling)
- **LLM Latency**: Variable (1-30 seconds depending on model, 10-100x faster with cache hits)
- **Parallel Scaling**: Near-linear with worker count, enhanced by shared caching
- **Total Speedup**: 60-70% reduction in execution time vs. original implementation

### Resource Requirements
- **Memory**: ~2GB base + 500MB per parallel worker + container pool overhead
- **Storage**: ~1GB for container images, ~100MB per test run, ~50MB for cache files
- **Network**: Model-dependent (100KB-10MB per test, greatly reduced with caching)

### Multi-Layer Caching Architecture

The framework implements a sophisticated three-tier caching system designed for performance, reproducibility, and scientific rigor:

#### 1. LLM Response Caching (`llm_cache.py`) - **Dedicated Cache Architecture**
**Purpose**: Eliminate expensive API calls for repeated prompts across all worker processes  
**Location**: `tmp/cache-{model_name}.p`  
**Architecture**: Extracted to dedicated `LLMCache` class with clean interface

**Key Features**:
- **Backward-compatible keys**: Supports both legacy and new cache key formats
- **Model-specific storage**: Each model maintains separate cache files  
- **Persistent across runs**: Survives system restarts and script re-execution
- **Multiprocess-safe**: File locking prevents corruption during concurrent access
- **Atomic writes**: Prevents partial cache corruption during parallel updates
- **Auto-migration**: Legacy cache entries automatically migrate to new format
- **Robust error handling**: Graceful handling of corrupted cache files

**Enhanced Cache Management**:
- **Dedicated Class**: `LLMCache` class with `.get()`, `.put()`, `.get_cache_key()` interface
- **Smart Fallback**: Try new format → Try legacy format → Auto-migrate on hit
- **Empty Cache Detection**: Forces reload if cache empty but file exists (fixes worker startup issues)
- **File Locking Strategy**: Shared locks for concurrent reads, exclusive locks for writes
- **Atomic Write Pattern**: Write to temporary file, then atomic rename to prevent corruption
- **Test Coverage**: Comprehensive unit tests ensure cache reliability

#### 2. Test Result Caching (`main.py:534-547`)
**Purpose**: Enable cross-commit comparison and rapid report generation  
**Location**: `results/{git_commit_hash}/{model}-run{N}.p`  
**Key Features**:
- **Git-based versioning**: Results organized by commit hash for reproducibility
- **Multi-run aggregation**: Support for multiple runs per model/commit
- **Complete execution traces**: Full dataflow pipeline results with Reason objects
- **Cross-temporal analysis**: Compare model performance across code versions

**Storage Structure**:
```python
# File naming convention
f"{logdir}/{commit_hash}/{model}-run{run_id}.p"

# Data format per file
{
    "test_file.py.TestClassName": (success_boolean, reason_tree),
    "print_hello.py.TestPrintHello": (True, execution_trace),
    "convert_to_c.py.TestRewriteC": (False, failure_trace)
}
```

#### 3. HTML Report Caching (`evaluation_examples/`)
**Purpose**: Pre-generated interactive reports for fast web browsing  
**Location**: `evaluation_examples/*.html`  
**Key Features**:
- **Syntax highlighting**: Pygments-based code formatting with language detection
- **Interactive filtering**: JavaScript-based tag filtering and sorting
- **Visual asset management**: PNG files for image-based test outputs
- **Drill-down navigation**: From overview grids to detailed execution traces
- **Responsive design**: Optimized for both desktop and mobile viewing

**Report Hierarchy**:
```
evaluation_examples/
├── index.html                           # Main comparison grid
├── {test_name}.html                      # Per-test overview pages
├── {test}.py.{TestClass}_{model}.html    # Detailed execution traces
└── {test}_{model}_{run_id}_{suffix}.png  # Visual test outputs
```

### Cache Invalidation Strategy

**LLM Cache**: Manual invalidation only - delete `tmp/cache-*.p` files
**Result Cache**: Automatic via git commits - new commits create new cache directories  
**HTML Cache**: On-demand regeneration with `--generate-report` flag

**Performance Impact**:
- **LLM caching**: 10-100x speedup for repeated evaluations, eliminates API costs
- **Result caching**: Near-instantaneous report generation from cached data
- **HTML caching**: Sub-second report loading for large datasets

### Cache Consistency Guarantees

**LLM Cache Consistency**: 
- Cache keys include hyperparameters ensuring configuration changes invalidate cache
- **Multiprocess Safety**: File locking prevents race conditions during concurrent access
- **Atomic Operations**: Temp file + rename pattern prevents partial writes
- **Cross-Process Visibility**: Workers reload cache to see each other's entries

**Result Cache Integrity**: Git commit hashing ensures results can never be mixed between code versions
**HTML Cache Accuracy**: Generated deterministically from result cache, ensuring consistency

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

### Completed Optimizations ✅
- **Container Pooling**: ✅ **Implemented** - Reusable warm containers with thread-safe queue management
- **Parallel Caching**: ✅ **Implemented** - Multiprocess-safe LLM response caching with file locking

### Scalability Improvements
- **Distributed Execution**: Multi-machine test distribution across network
- **Result Streaming**: Real-time result updates via WebSocket or SSE
- **Dynamic Pool Sizing**: Auto-adjust container pool based on system load

### Enhanced Security
- **SELinux/AppArmor**: Additional container hardening
- **Network Policies**: Fine-grained network restrictions
- **Runtime Monitoring**: Container behavior analysis

### Advanced Evaluation
- **Multi-modal Testing**: Code + documentation + images
- **Interactive Testing**: Multi-turn debugging scenarios
- **Performance Benchmarking**: Execution time/memory measurement

This architecture enables systematic evaluation of LLM capabilities on realistic programming tasks while maintaining security, reproducibility, and extensibility.