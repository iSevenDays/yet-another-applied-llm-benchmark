## Copyright (C) 2024, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

import sys
import re
import importlib
import tests
import os
import llm
import json
import argparse
import pickle
import subprocess
import time
from tqdm import tqdm
import create_results_html
import traceback
import logging
import docker

from evaluator import Env, Conversation, run_test

import multiprocessing as mp
from functools import partial

# Configuration constants
MAX_FAILURE_REASON_LENGTH = 200
MAX_WAIT_TIME_SECONDS = 900  # 15 minutes
STATUS_LOG_INTERVAL_SECONDS = 30
WORKER_LOG_CHUNK_INTERVAL = 500
STAGE_HANG_THRESHOLD_SECONDS = 60
DEFAULT_MODELS = [
    "gpt-4o", "gpt-4-0125-preview", "claude-3-opus-20240229", 
    "claude-3-sonnet-20240229", "gpt-3.5-turbo-0125", "gemini-pro", 
    "mistral-large-latest", "mistral-medium"
]

def _load_test_module(filename):
    """Load a test module and return module object and test names.
    
    Returns:
        tuple: (module, test_names_list) or (None, []) if loading fails
    """
    module_name = filename[:-3]
    file_path = os.path.join("tests", filename)
    
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if not spec or not spec.loader:
            logging.warning(f"Could not create spec for {filename}. Skipping.")
            return None, []
            
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        
        test_names_in_file = [
            name for name in dir(module) 
            if name.startswith("Test") and hasattr(getattr(module, name), '__call__')
        ]
        
        return module, test_names_in_file
        
    except Exception as import_exc:
        logging.error(f"Failed to load test module {filename}: {import_exc}")
        if module_name in sys.modules:
            del sys.modules[module_name]
        return None, []

def _discover_test_data_for_parallel(test_files, pbar, test_llm):
    """Discover all test data needed for parallel execution.
    
    Args:
        test_files: List of test file names
        pbar: Progress bar instance
        test_llm: Test LLM instance
    
    Returns:
        list: Test data tuples for parallel execution
    """
    all_test_data = []
    
    for filename in test_files:
        module, test_names_in_file = _load_test_module(filename)
        
        if not module or not test_names_in_file:
            if not module:
                pbar.write(f"Warning: Could not load module {filename}. Skipping.")
            continue
            
        for test_name in test_names_in_file:
            # Check if this test requires vision LLM and skip if not available
            test_instance = getattr(module, test_name)
            if requires_vision_llm(test_instance) and llm.vision_eval_llm is None:
                pbar.write(f"SKIP: {filename}.{test_name} (requires vision LLM but none configured)")
                continue
                
            # Prepare test data for multiprocessing
            eval_llm_name = llm.eval_llm.original_name if llm.eval_llm else None
            vision_eval_llm_name = llm.vision_eval_llm.original_name if llm.vision_eval_llm else None
            file_path = os.path.join("tests", filename)
            
            test_data = (
                f"{filename}.{test_name}", 
                file_path, 
                test_name, 
                test_llm.original_name if hasattr(test_llm, 'original_name') else str(test_llm),
                eval_llm_name, 
                vision_eval_llm_name
            )
            all_test_data.append(test_data)
            
    return all_test_data

def _process_test_in_sequential_mode(filename, test_name, module, test_llm, sr, total_passed, total_failed, pbar):
    """Process a single test in sequential mode.
    
    Returns:
        bool: True if test passed, False otherwise
    """
    # Check if this test requires vision LLM and skip if not available
    test_instance = getattr(module, test_name)
    if requires_vision_llm(test_instance) and llm.vision_eval_llm is None:
        pbar.write(f"SKIP: {test_name} (requires vision LLM but none configured)")
        return False
    
    # Show current test and running totals
    total_tests = total_passed[0] + total_failed[0]
    pass_rate = (total_passed[0] / total_tests * 100) if total_tests > 0 else 0
    pbar.set_description_str(f"Running: {test_name} ({total_passed[0]}✓/{total_failed[0]}✗/{total_tests}, {pass_rate:.1f}%)")
    pbar.refresh()

    # Redirect stdout during test execution
    original_stdout = sys.stdout
    devnull = open(os.devnull, 'w')
    test_passed = False
    failure_reason = "Test execution did not yield result"
    
    try:
        sys.stdout = devnull
        test_passed, failure_reason = run_one_test(
            test_instance,
            test_llm,
            llm.eval_llm,
            llm.vision_eval_llm
        )
    except Exception as run_exc:
        test_passed = False
        failure_reason = f"Unhandled exception in test run: {run_exc}"
        pbar.write("\n--- Unhandled Exception Traceback ---")
        traceback.print_exc(file=sys.stderr)
        pbar.write("--- End Traceback ---")
    finally:
        sys.stdout = original_stdout
        devnull.close()

    # Update counters and report results
    if test_passed:
        total_passed[0] += 1
    else:
        total_failed[0] += 1
    
    total_tests = total_passed[0] + total_failed[0]
    result_str = "PASS" if test_passed else "FAIL"
    pbar.write(f"{result_str}: {test_name}")
    if not test_passed:
        reason_str = format_failure_reason(failure_reason)
        pbar.write(f"  Reason: {reason_str}")

    sr[f"{filename}.{test_name}"] = (test_passed, failure_reason)
    return test_passed

class ConciseFormatter(logging.Formatter):
    """Custom formatter for abbreviated log levels."""
    level_map = {
        'WARNING': 'WARN',
        'CRITICAL': 'CRIT',
        'ERROR': 'ERR',
        'INFO': 'INFO',
        'DEBUG': 'DBUG'
    }
    
    def format(self, record):
        # Abbreviate level name
        original_levelname = record.levelname
        record.levelname = self.level_map.get(original_levelname, original_levelname)
        result = super().format(record)
        # Restore original level name
        record.levelname = original_levelname
        return result

def _setup_worker_logging():
    """Configure efficient logging for worker processes following SPARC principles."""
    import threading
    import sys
    import time
    
    logger = logging.getLogger()
    logger.handlers.clear()
    
    try:
        # SPARC: Simple approach - use WARNING level in workers to reduce I/O overhead
        # Critical errors still logged, but debug noise eliminated in parallel mode
        worker_log_level = logging.WARNING
        
        # Only create file handler if DEBUG level is explicitly needed
        if os.environ.get('BENCHMARK_DEBUG_WORKERS') == '1':
            # Generate timestamp-based filename
            timestamp = time.strftime('%Y-%m-%d-%H-%M-%S')
            worker_log_file = f'logs/debug_stream_worker_{timestamp}.log'
            file_handler = logging.FileHandler(worker_log_file, mode='a')
            file_handler.setLevel(logging.DEBUG)
            # Concise format: shortened timestamp, abbreviated level, abbreviated function name
            formatter = ConciseFormatter('%(asctime)s,%(msecs)01d - %(levelname)s - [%(funcName)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            worker_log_level = logging.DEBUG
        else:
            # SPARC: Focus on essential logging only - use stderr for critical worker issues
            console_handler = logging.StreamHandler(sys.stderr)
            console_handler.setLevel(logging.WARNING)
            formatter = ConciseFormatter('WORKER-%(levelname)s: %(message)s')
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        logger.setLevel(worker_log_level)
        logger.propagate = False
        
        # Disable noisy third-party loggers
        for logger_name in ['httpcore', 'httpx', 'openai._base_client', 'httpcore.connection', 'httpcore.http11']:
            httpcore_logger = logging.getLogger(logger_name)
            httpcore_logger.disabled = True
            httpcore_logger.handlers.clear()
            httpcore_logger.propagate = False
    except Exception as log_setup_error:
        print(f"Worker logging setup failed: {log_setup_error}")
        logging.basicConfig(level=logging.WARNING)

def format_failure_reason(failure_reason, max_length=MAX_FAILURE_REASON_LENGTH):
    """Extract readable failure message from Reason object or string"""
    
    def find_evaluator_failure(reason):
        """Recursively find the deepest evaluator node that failed"""
        if not hasattr(reason, 'node') or not hasattr(reason, 'children'):
            return None
            
        node_name = reason.node.__name__ if hasattr(reason.node, '__name__') else str(reason.node)
        
        # If this is an evaluator node, check if it failed
        if 'Evaluator' in node_name and len(reason.children) >= 2:
            expected, result = reason.children[0], reason.children[1]
            if result is False:  # This evaluator failed
                return f"{node_name}: Expected '{expected}' but test failed"
        
        # Recursively search children for evaluator failures
        if isinstance(reason.children, (list, tuple)):
            for child in reason.children:
                if hasattr(child, 'node'):
                    found = find_evaluator_failure(child)
                    if found:
                        return found
        
        return None
    
    if hasattr(failure_reason, 'node'):
        evaluator_failure = find_evaluator_failure(failure_reason)
        if evaluator_failure:
            reason_str = evaluator_failure
        else:
            node_name = failure_reason.node.__name__ if hasattr(failure_reason.node, '__name__') else str(failure_reason.node)
            reason_str = f"Test failed at {node_name} step"
    else:
        reason_str = str(failure_reason)
    
    if len(reason_str) > max_length: 
        reason_str = reason_str[:max_length-3] + "..."
    return reason_str

def requires_vision_llm(test_node):
    """
    Recursively check if a test node tree contains LLMVisionRun nodes.
    Returns True if the test requires vision LLM, False otherwise.
    """
    from evaluator import LLMVisionRun
    
    if isinstance(test_node, LLMVisionRun):
        return True
    
    # Check binary nodes (ThenNode, AndNode, OrNode)
    if hasattr(test_node, 'node1') and hasattr(test_node, 'node2'):
        return requires_vision_llm(test_node.node1) or requires_vision_llm(test_node.node2)
    
    # Check unary nodes (NotNode)
    if hasattr(test_node, 'node'):
        return requires_vision_llm(test_node.node)
    
    # Check UntilDone node
    if hasattr(test_node, 'cond') and hasattr(test_node, 'body'):
        return requires_vision_llm(test_node.cond) or requires_vision_llm(test_node.body)
    
    return False

def run_test_with_name(test_data):
    """
    Wrapper function for multiprocessing that includes test name.
    test_data: (test_name, file_path, test_class_name, test_llm_name, eval_llm_name, vision_eval_llm_name)
    Returns: (test_name, test_passed, failure_reason)
    """
    test_name, file_path, test_class_name, test_llm_name, eval_llm_name, vision_eval_llm_name = test_data
    try:
        # Import necessary modules in worker process
        import sys
        import os
        import importlib.util
        import logging
        
        _setup_worker_logging()
        
        # Log worker start with execution timestamp for hang detection
        execution_start = time.time()
        logging.info(f"Worker process started for test: {test_name} EXEC_START:{execution_start}")
        
        # Add current directory to Python path so modules can be imported
        sys.path.insert(0, os.getcwd())
        
        # Load the test module
        module_name = os.path.basename(file_path)[:-3]  # Remove .py extension
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        
        # Get the test class
        test_instance = getattr(module, test_class_name)
        
        # Recreate LLM instances in worker process with caching enabled for performance
        import llm
        test_llm = llm.LLM(test_llm_name, use_cache=True)
        eval_llm = llm.LLM(eval_llm_name, use_cache=True) if eval_llm_name else None
        vision_eval_llm = llm.LLM(vision_eval_llm_name, use_cache=True) if vision_eval_llm_name else None
        
        test_passed, failure_reason = run_one_test(test_instance, test_llm, eval_llm, vision_eval_llm)
        return (test_name, test_passed, failure_reason)
    except Exception as e:
        import traceback
        error_msg = f"Exception in test execution: {e}"
        traceback_str = traceback.format_exc()
        
        # Log the full error details for debugging
        logging.error(f"Test {test_name} failed with exception: {error_msg}")
        logging.error(f"Traceback: {traceback_str}")
        
        # Print a shorter version to stdout so user can see it
        print(f"EXCEPTION {str(e)}")
        
        return (test_name, False, error_msg)

def _cleanup_container(env):
    """Safely cleanup container resources using pool when available."""
    import docker_controller
    if env.container:
        if hasattr(docker_controller, 'return_container_to_pool'):
            docker_controller.return_container_to_pool(env)
        else:
            # Fallback for compatibility
            docker_controller.async_kill_container(env.docker, env.container)

def _handle_docker_error(docker_err):
    """Format Docker error messages for user display."""
    err_str = str(docker_err).lower()
    connection_errors = ["connection aborted", "file not found", "connection refused", "docker daemon"]
    
    if any(error in err_str for error in connection_errors):
        return "Failed to connect to Docker/Podman. Please ensure the daemon/service is running and accessible."
    return f"Docker error: {docker_err}"

def run_one_test(test, test_llm, eval_llm, vision_eval_llm):
    """
    Runs just one test case and returns either true or false and the output.
    """
    import time
    
    test_name = getattr(test, '__name__', str(test))
    logging.info(f"STAGE: Starting test execution for {test_name}")
    
    env = Env()
    test.setup(env, Conversation(test_llm), test_llm, eval_llm, vision_eval_llm)

    try:
        output = "No test output generated"
        stage_count = 0
        last_stage_time = time.time()
        
        logging.info(f"STAGE: Entering test pipeline for {test_name}")
        for success, output in test():
            stage_count += 1
            current_time = time.time()
            stage_duration = current_time - last_stage_time
            
            logging.info(f"STAGE: {test_name} stage {stage_count} completed in {stage_duration:.1f}s, success={success}")
            
            if stage_duration > STAGE_HANG_THRESHOLD_SECONDS:
                logging.warning(f"HANG_DETECT: {test_name} stage {stage_count} took {stage_duration:.1f}s (>{STAGE_HANG_THRESHOLD_SECONDS}s)")
            
            last_stage_time = current_time
            
            if success:
                logging.info(f"STAGE: {test_name} completed successfully after {stage_count} stages")
                _cleanup_container(env)
                return True, output
                
        logging.info(f"STAGE: {test_name} failed after {stage_count} stages")
        _cleanup_container(env)
        return False, output
        
    except GeneratorExit:
        _cleanup_container(env)
        return False, "Test was interrupted"
    except docker.errors.DockerException as docker_err:
        _cleanup_container(env)
        return False, _handle_docker_error(docker_err)
    except Exception as e:
        _cleanup_container(env)
        return False, f"An error occurred: {str(e)}"
                    

def _run_tests_sequential(test_files, pbar, test_llm, sr, total_passed, total_failed):
    """Sequential test execution using shared test discovery logic."""
    for filename in test_files:
        module, test_names_in_file = _load_test_module(filename)
        
        if not module:
            pbar.write(f"Warning: Could not load module {filename}. Skipping.")
            pbar.update(1)
            continue
            
        if not test_names_in_file:
            pbar.write(f"Info: No tests found in {filename}.")
            pbar.update(1)
            continue

        # Process tests in this file
        num_tests_in_file = len(test_names_in_file)
        initial_file_progress = pbar.n

        for i, test_name in enumerate(test_names_in_file):
            _process_test_in_sequential_mode(
                filename, test_name, module, test_llm, sr, total_passed, total_failed, pbar
            )

            # Update progress bar
            progress_increment = (1.0 / num_tests_in_file)
            current_progress = min(initial_file_progress + (i + 1) * progress_increment, initial_file_progress + 1)
            pbar.n = max(current_progress, pbar.n)
            
            # Update description with running totals
            total_tests = total_passed[0] + total_failed[0]
            pass_rate = (total_passed[0] / total_tests * 100) if total_tests > 0 else 0
            pbar.set_description_str(f"Files ({total_passed[0]}✓/{total_failed[0]}✗/{total_tests} tests, {pass_rate:.1f}%)")
            pbar.refresh()

        # Ensure the progress bar cleanly reaches the next integer after file completion
        pbar.n = initial_file_progress + 1
        pbar.refresh()

def _check_completed_jobs(async_results, completed, job_start_times, job_test_names, 
                         current_time, total_passed, total_failed, sr, pbar):
    """Check for completed jobs and process their results. Returns True if any job completed."""
    found_completion = False
    
    for i, async_result in enumerate(async_results):
        if i not in completed:
            try:
                if async_result.ready():
                    test_name, test_passed, failure_reason = async_result.get(timeout=5.0)
                    completed.append(i)
                    found_completion = True
                    duration = current_time - job_start_times[i]
                    logging.info(f"PARALLEL: Job {test_name} completed in {duration:.1f}s, success={test_passed}")
                    
                    # Process the result
                    if test_passed:
                        total_passed[0] += 1
                    else:
                        total_failed[0] += 1
                        
                    sr[test_name] = (test_passed, failure_reason)
                    
                    result_str = "PASS" if test_passed else "FAIL"
                    pbar.write(f"{result_str}: {test_name}")
                    if not test_passed:
                        reason_str = format_failure_reason(failure_reason)
                        pbar.write(f"  Reason: {reason_str}")
                        
                    pbar.update(1)
                    
                    # Update description with running totals
                    total_tests = total_passed[0] + total_failed[0]
                    pass_rate = (total_passed[0] / total_tests * 100) if total_tests > 0 else 0
                    pbar.set_description_str(f"Parallel ({total_passed[0]}✓/{total_failed[0]}✗/{total_tests} tests, {pass_rate:.1f}%)")
                    pbar.refresh()
                else:
                    # Only log when we're waiting for the last few tests (reduced frequency)
                    if len(completed) >= len(async_results) - 2:
                        logging.info(f"PARALLEL: Waiting for {job_test_names[i]} (completed: {len(completed)}/{len(async_results)})")
                        
            except Exception as e:
                logging.error(f"PARALLEL: Error checking/getting result for job {i} ({job_test_names[i]}): {e}")
                completed.append(i)  # Mark as completed to avoid infinite loop
                found_completion = True
    
    return found_completion

def _run_tests_parallel(test_files, pbar, test_llm, sr, total_passed, total_failed, parallel_workers):
    """Parallel test execution using shared test discovery logic."""
    # Use shared test discovery logic
    all_test_data = _discover_test_data_for_parallel(test_files, pbar, test_llm)
    
    if not all_test_data:
        pbar.write("No tests to run")
        return
    
    # Update progress bar total to reflect actual number of tests
    pbar.total = len(all_test_data)
    pbar.set_description_str(f"Running {len(all_test_data)} tests with {parallel_workers} workers")
    pbar.refresh()
    
    # Run tests in parallel with real-time progress updates and hanging detection
    with mp.Pool(parallel_workers) as pool:
        # Submit all jobs asynchronously
        async_results = []
        job_start_times = {}
        job_test_names = {}
        
        for i, test_data in enumerate(all_test_data):
            result = pool.apply_async(run_test_with_name, (test_data,))
            async_results.append(result)
            job_start_times[i] = time.time()
            job_test_names[i] = test_data[0]  # test_name is first element
            
        logging.info(f"PARALLEL: Started {len(async_results)} jobs with {parallel_workers} workers")
        
        # Process results as they complete with hanging detection
        completed = []
        last_status_log = time.time()
        
        while len(completed) < len(async_results):
            current_time = time.time()
            
            # Log progress for debugging last test hangs
            if len(completed) >= len(async_results) - 1:  # Last test
                remaining = [i for i in range(len(async_results)) if i not in completed]
                logging.info(f"PARALLEL: Waiting for last {len(remaining)} jobs: {[job_test_names[i] for i in remaining]}")
            
            # Periodic status logging
            if current_time - last_status_log > STATUS_LOG_INTERVAL_SECONDS:
                running_jobs = []
                for i, result in enumerate(async_results):
                    if i not in completed and not result.ready():
                        duration = current_time - job_start_times[i]
                        running_jobs.append(f"{job_test_names[i]}({duration:.0f}s)")
                
                if running_jobs:
                    logging.info(f"PARALLEL: {len(running_jobs)} jobs still running: {', '.join(running_jobs)}")
                last_status_log = current_time
            
            # Check for completed jobs
            found_completion = _check_completed_jobs(async_results, completed, job_start_times, 
                                                   job_test_names, current_time, total_passed, 
                                                   total_failed, sr, pbar)
            
            # Small sleep to avoid busy waiting
            time.sleep(0.1)
        

def run_all_tests(test_llm_name, use_cache=True, which_tests=None, parallel_workers=1):
    """
    Run every test case in the benchmark, returning a dictionary of the results
    of the format { "test_name": (success, output) }
    """
    test_llm = llm.LLM(test_llm_name, use_cache=use_cache)
    print(f'test_llm: {test_llm.name}')
    print(f'llm.eval_llm: {llm.eval_llm.name}')
    sr = {}
    
    # Get the list of test files
    test_files = [f for f in os.listdir("tests") if f.endswith(".py")]
    if which_tests is not None:
        test_files = [f for f in test_files if f[:-3] in which_tests]
    
    # Create a tqdm progress bar with pass/fail tracking
    total_passed = [0]  # Use list for mutable reference
    total_failed = [0]  # Use list for mutable reference
    with tqdm(total=len(test_files), desc="Files", unit="file", ascii=" >==") as pbar:
        
        if parallel_workers == 1:
            # Sequential execution (preserves existing behavior)
            _run_tests_sequential(test_files, pbar, test_llm, sr, total_passed, total_failed)
        else:
            # Parallel execution
            _run_tests_parallel(test_files, pbar, test_llm, sr, total_passed, total_failed, parallel_workers)
    
    # Final summary
    total_tests = total_passed[0] + total_failed[0]
    pass_rate = (total_passed[0] / total_tests * 100) if total_tests > 0 else 0
    print(f"\n🎯 Final Results: {total_passed[0]} passed, {total_failed[0]} failed out of {total_tests} tests ({pass_rate:.1f}% pass rate)")
    
    return sr


def get_tags():
    """
    Each test has a description and a set of tags. This returns dictionaries
    of the format { "test_name": "description" } and { "test_name": ["tag1", "tag2"] }
    """
    descriptions = {}
    tags = {}
    for f in os.listdir("tests"):
        if not f.endswith(".py"): continue
        try:
            spec = importlib.util.spec_from_file_location(f[:-3], "tests/" + f)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except (ImportError, AttributeError, SyntaxError) as e:
            print(f"Warning: Skipping {f} due to import error: {e}")
            continue
        if 'TAGS' in dir(module):
            test_case = [x for x in dir(module) if x.startswith("Test") and x != "TestCase"]
            for t in test_case:
                tags[f+"."+t] = module.TAGS
                descriptions[f+"."+t] = module.DESCRIPTION
    return tags, descriptions

def discover_available_models(logdir):
    """
    Discover all available models from saved runs across all commits.
    
    Args:
        logdir: Directory containing saved runs organized by commit hash
        
    Returns:
        List of unique model names found across all commits, sorted alphabetically
    """
    import re
    all_models = set()
    
    try:
        commit_hashes = get_ordered_logs(logdir)
        
        for commit_hash in commit_hashes:
            commit_dir = os.path.join(logdir, commit_hash)
            if os.path.exists(commit_dir):
                for filename in os.listdir(commit_dir):
                    # Extract model name from filename pattern: {model_name}-run{N}.p
                    match = re.match(r"^(.+)-run\d+\.p$", filename)
                    if match:
                        model_name = match.group(1)
                        all_models.add(model_name)
                        
    except Exception as e:
        logging.warning(f"Error discovering models: {e}")
        return []
    
    return sorted(list(all_models))

def get_ordered_logs(logdir):
    hashes = []
    for githash in os.listdir(logdir):
        if '-run' in githash:
            print("There was a breaking change in how results are stored. Please move the runs into args.logdir/[git commit hash]/[the results].")
            exit(1)
        hashes.append(githash)
    
    command = ['git', 'log', '--pretty=format:%H']
    result = subprocess.run(command, capture_output=True, text=True)
    commit_hashes = result.stdout.strip().split('\n')
    commit_hashes = [x for x in commit_hashes if x in hashes]
    return commit_hashes

def load_saved_runs(output_dir, model):
    """
    Load saved runs from the output directory for a specific model.
    """
    saved_runs = {}
    for file in sorted(os.listdir(output_dir)):
        if file.startswith(model+"-run"):
            one_run = None
            if '.json' in file:
                with open(os.path.join(output_dir, file), 'r') as f:
                    one_run = json.loads(f.readlines()[-1])
            elif '.p' in file:
                one_run = pickle.load(open(os.path.join(output_dir, file), 'rb'))
            try:
                for k,(v1,v2) in one_run.items():
                    if k not in saved_runs:
                        saved_runs[k] = ([], [])
                    saved_runs[k][0].append(v1)
                    saved_runs[k][1].append(v2)
            except json.JSONDecodeError:
                print(f"Warning: Invalid JSON in file {file}")
    return saved_runs

def _setup_main_logging():
    """Configure efficient main process logging following SPARC principles."""
    import time
    
    # Generate timestamp-based filename
    timestamp = time.strftime('%Y-%m-%d-%H-%M-%S')
    log_filename = f'logs/debug_stream_main_{timestamp}.log'
    
    # SPARC: Simple configuration with performance-conscious defaults
    console_level = logging.INFO  # Reduce console noise
    file_level = logging.DEBUG if os.environ.get('BENCHMARK_VERBOSE_LOGS') == '1' else logging.INFO
    
    # Create handlers with custom formatter
    file_handler = logging.FileHandler(log_filename, mode='a')
    console_handler = logging.StreamHandler(sys.stdout)
    
    # Apply concise formatter
    formatter = ConciseFormatter('%(asctime)s,%(msecs)01d - %(levelname)s - [%(funcName)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Configure logging
    logging.basicConfig(
        level=file_level,
        handlers=[file_handler, console_handler]
    )
    # Set console handler to reduced level for cleaner output
    console_handler.setLevel(console_level)
    
    # Only log initialization if verbose mode is enabled
    if file_level == logging.DEBUG:
        logging.info(f"Verbose logging enabled. DEBUG logs going to {log_filename}")
    else:
        logging.info(f"Efficient logging active. INFO+ logs in {log_filename}")

def main():
    _setup_main_logging()

    parser = argparse.ArgumentParser(description="Run tests on language models.")
    parser.add_argument('--model', help='Specify a specific model to run.', type=str, action="append")
    parser.add_argument('--all-models', help='Run all models.', action='store_true')
    parser.add_argument('--add-all', help='Auto-discover and load all previously saved models.', action='store_true')
    parser.add_argument('--add', help='Load all previously saved models plus specified models.', action='store_true')
    
    parser.add_argument('--test', help='Specify a specific test to run.', type=str, action="append")
    
    parser.add_argument('--times', help='Number of times to run the model(s).', type=int, default=1)
    parser.add_argument('--runid', help='Offset of the run ID for saving.', type=int, default=0)
    parser.add_argument('--logdir', help='Output path for the results.', type=str, default='results')
    parser.add_argument('--generate-report', help='Generate an HTML report.', action='store_true')
    parser.add_argument('--load-saved', help='Load saved evaluations.', action='store_true')
    parser.add_argument('--run-tests', help='Run a batch of tests.', action='store_true')
    parser.add_argument('--parallel', help='Number of parallel workers for test execution. Default: 1 (sequential)', type=int, default=1)
    parser.add_argument('--only-changed', help='Only run tests that have changed since the given commit (INCLUSIVE).')

    args = parser.parse_args()

    assert args.run_tests ^ args.load_saved, "Exactly one of --run-tests or --load-saved must be specified."
    
    # Validate model selection arguments - following SPARC principle of clear validation
    # Note: --add can be combined with --model, but others are mutually exclusive
    exclusive_args = [args.all_models, args.add_all]
    if args.add:
        # --add can work with --model, but not with other flags
        if any(exclusive_args):
            parser.error("Cannot combine --add with --all-models or --add-all.")
    else:
        # Without --add, check for mutual exclusivity
        model_selection_args = exclusive_args + [bool(args.model)]
        if sum(model_selection_args) > 1:
            parser.error("Cannot combine --all-models, --add-all, and --model. Choose one approach.")
    
    # Validate that --add and --add-all require --load-saved
    if (args.add or args.add_all) and not args.load_saved:
        parser.error("--add and --add-all can only be used with --load-saved.")
    
    # Create the results directory if it doesn't exist
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    # Model selection logic - enhanced following SPARC iteration principle
    models_to_run = []
    if args.add_all:
        models_to_run = discover_available_models(args.logdir)
        print(f"Auto-discovered {len(models_to_run)} models: {', '.join(models_to_run)}")
    elif args.add:
        discovered = discover_available_models(args.logdir)
        specified = args.model or []
        models_to_run = sorted(list(set(discovered + specified)))
        print(f"Loading {len(discovered)} discovered + {len(specified)} specified models")
        if discovered:
            print(f"  Discovered: {', '.join(discovered)}")
        if specified:
            print(f"  Specified: {', '.join(specified)}")
    elif args.model:
        models_to_run = args.model
    elif args.all_models:
        models_to_run = DEFAULT_MODELS

    data = {}
    for model in models_to_run:
        if args.load_saved:
            data[model] = {}
            commit_hashes = get_ordered_logs(args.logdir)
            print("Loading data from commits")

            for githash in commit_hashes[::-1]:
                print(githash)
                kvs = load_saved_runs(os.path.join(args.logdir, githash), model)
                for k,v in kvs.items():
                    data[model][k] = v
        elif args.run_tests:
            tests_subset = None # run all of them
            
            if args.test:
                tests_subset = args.test # run the ones the user said
            elif args.only_changed:
                latest_commit_finished = args.only_changed
                command = ['git', 'diff', '--name-only', latest_commit_finished+"^", 'HEAD']
                
                result = subprocess.run(command, capture_output=True, text=True)
                changed_files = result.stdout.strip().split('\n')
                changed_files = [x.split("tests/")[1].split(".py")[0] for x in changed_files if x.startswith("tests/")]
                print("Running the following tests:\n  -",
                      "\n  - ".join(changed_files))
                tests_subset = set(changed_files)

            
            command = ['git', 'rev-parse', 'HEAD']
            result = subprocess.run(command, capture_output=True, text=True)
            current_commit_hash = result.stdout.strip()

            data[model] = {}
            for i in range(args.times):
                print(f"Running {model}, iteration {i+args.runid}")
                result = run_all_tests(model, use_cache=True,
                                       which_tests=tests_subset,
                                       parallel_workers=args.parallel)

                for k,(v1,v2) in result.items():
                    if k not in data[model]:
                        data[model][k] = ([], [])
                    data[model][k][0].append(v1)
                    data[model][k][1].append(v2)

                if not os.path.exists(os.path.join(args.logdir, current_commit_hash)):
                    os.mkdir(os.path.join(args.logdir, current_commit_hash))
                with open(f"{args.logdir}/{current_commit_hash}/{model}-run{i+args.runid}.p", 'wb') as f:
                    pickle.dump(result, f)
        else:
            raise RuntimeError("Unreachable code path - invalid execution mode")

    if args.generate_report:
        tags, descriptions = get_tags()  # Assuming these functions are defined in your codebase
        create_results_html.generate_report(data, tags, descriptions)

if __name__ == "__main__":
    main()
