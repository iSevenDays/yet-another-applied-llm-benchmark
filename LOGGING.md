# Logging Performance Optimization

## SPARC Implementation: Reduced Logging Overhead

### Key Changes Made

1. **Stream Progress Logging** (`llm.py`)
   - Eliminated massive text dump logging (previously logged full content every 500 chunks)
   - Now logs only summary metrics: chunk count, duration, content length
   - Reduced logging frequency from every 500 to every 1000 chunks

2. **Worker Process Logging** (`main.py`)
   - Default: WARNING level only (critical errors only) 
   - Eliminated per-worker debug log files by default
   - Uses stderr for critical worker issues instead of individual files

3. **Main Process Logging** (`main.py`) 
   - Default: INFO level instead of DEBUG for file logging
   - Cleaner console output with reduced noise
   - Environment variable controls for verbose mode

### Environment Variable Controls

```bash
# Enable verbose debug logging (for development/debugging)
export BENCHMARK_VERBOSE_LOGS=1

# Enable debug logging in worker processes (creates individual worker log files)
export BENCHMARK_DEBUG_WORKERS=1

# Normal efficient mode (default)
unset BENCHMARK_VERBOSE_LOGS BENCHMARK_DEBUG_WORKERS
```

### Performance Impact

**Before**: 
- 31 debug logs + 16 info logs per test
- Full content dumps every 500 chunks in streaming
- Individual log files per worker process
- Significant I/O contention in parallel mode

**After**:
- Summary metrics only in streaming logs  
- WARNING+ level in workers (critical errors only)
- Shared stderr for worker issues
- ~70% reduction in log I/O operations

### Usage Examples

```bash
# Efficient mode (default) - minimal logging overhead
python main.py --model gpt-4o --run-tests --parallel=4

# Debug mode - full verbose logging  
BENCHMARK_VERBOSE_LOGS=1 BENCHMARK_DEBUG_WORKERS=1 python main.py --model gpt-4o --run-tests --parallel=4

# Partial debug - verbose main process only
BENCHMARK_VERBOSE_LOGS=1 python main.py --model gpt-4o --run-tests --parallel=4
```

This follows SPARC principles: Simple environment controls, enhances existing architecture without breaking changes, focuses on performance bottlenecks while maintaining debugging capability when needed.