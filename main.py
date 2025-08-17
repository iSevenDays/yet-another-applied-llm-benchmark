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
from tqdm import tqdm
import create_results_html
import traceback
import logging
import docker # Added for exception handling

from evaluator import Env, Conversation, run_test

import multiprocessing as mp
from functools import partial

def format_failure_reason(failure_reason, max_length=200):
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
        
        # Worker process logging setup - force fresh configuration
        logger = logging.getLogger()
        # Clear any existing handlers to avoid conflicts
        logger.handlers.clear()
        
        # Set up dedicated worker logging (FILE ONLY - no console spam)
        file_handler = logging.FileHandler('debug_stream.log', mode='a')
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - WORKER - [%(funcName)s] %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False  # Prevent propagation to avoid console spam
        
        # Immediate test log to verify it works
        logging.info(f"Worker process started for test: {test_name}")
        
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
        
        # Recreate LLM instances in worker process (needed for multiprocessing)
        import llm
        test_llm = llm.LLM(test_llm_name, use_cache=False)
        eval_llm = llm.LLM(eval_llm_name, use_cache=False) if eval_llm_name else None
        vision_eval_llm = llm.LLM(vision_eval_llm_name, use_cache=False) if vision_eval_llm_name else None
        
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

def run_one_test(test, test_llm, eval_llm, vision_eval_llm):
    """
    Runs just one test case and returns either true or false and the output.
    """
    import docker_controller
    env = Env()
    test.setup(env, Conversation(test_llm), test_llm, eval_llm, vision_eval_llm)

    try:
        output = "No test output generated"  # Initialize output variable
        for success, output in test():
            if success:
                if env.container:
                    docker_controller.async_kill_container(env.docker, env.container)
                return True, output
        if env.container:
            docker_controller.async_kill_container(env.docker, env.container)
        return False, output
    except GeneratorExit:
        # Handle the case where the generator is closed prematurely
        if env.container:
            docker_controller.async_kill_container(env.docker, env.container)
        return False, "Test was interrupted"
    except docker.errors.DockerException as docker_err: # Catch Docker-specific errors
        # Check if it's likely a connection error
        err_str = str(docker_err).lower()
        user_message = f"Docker error: {docker_err}"
        if "connection aborted" in err_str or "file not found" in err_str or "connection refused" in err_str or "docker daemon" in err_str:
             user_message = "Failed to connect to Docker/Podman. Please ensure the daemon/service is running and accessible."
        if env.container:
            docker_controller.async_kill_container(env.docker, env.container)
        return False, user_message
    except Exception as e:
        # Handle any other exceptions that might occur
        if env.container:
            docker_controller.async_kill_container(env.docker, env.container)
        # Keep original formatting for other errors
        return False, f"An error occurred: {str(e)}"
                    

def _run_tests_sequential(test_files, pbar, test_llm, sr, total_passed, total_failed):
    """Sequential test execution - preserves original behavior exactly"""
    for filename in test_files:
        module = None # Ensure module is reset for each file
        module_name = filename[:-3]
        file_path = os.path.join("tests", filename)
        test_names_in_file = []

        try:
            # Load module and find tests for the current file (keep existing try/except logic here)
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if not spec or not spec.loader:
                pbar.write(f"Warning: Could not create spec for {filename}. Skipping.")
                pbar.update(1) # Ensure bar advances if file skipped
                continue

            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module # Handle relative imports
            spec.loader.exec_module(module)
            test_names_in_file = [name for name in dir(module) if name.startswith("Test") and hasattr(getattr(module, name), '__call__')]

        except Exception as import_exc:
            pbar.write(f"\nWarning: Skipping file {filename} due to import error:")
            for line in traceback.format_exception(import_exc):
                pbar.write(line.strip())
            if module_name in sys.modules:
                del sys.modules[module_name]
            pbar.update(1) # Ensure bar advances if file is skipped
            continue

        if not test_names_in_file:
            pbar.write(f"Info: No tests found in {filename}.")
            pbar.update(1) # Ensure bar advances if file has no tests
            continue

        # --- Start: Modified Inner Loop --- (Replaces the original inner loop)
        num_tests_in_file = len(test_names_in_file)
        initial_file_progress = pbar.n # Track current progress bar position

        for i, test_name in enumerate(test_names_in_file): # Loop per test
            # Show current test and running totals
            total_tests = total_passed[0] + total_failed[0]
            pass_rate = (total_passed[0] / total_tests * 100) if total_tests > 0 else 0
            pbar.set_description_str(f"Running: {test_name} ({total_passed[0]}✓/{total_failed[0]}✗/{total_tests}, {pass_rate:.1f}%)")
            pbar.refresh() # Ensure description updates immediately

            original_stdout = sys.stdout
            devnull = open(os.devnull, 'w') # Use os.devnull
            test_passed = False
            failure_reason = "Test execution did not yield result"
            try:
                sys.stdout = devnull # Redirect stdout
                test_instance = getattr(module, test_name)
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

            # Update pass/fail counters
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

            sr[f"{filename}.{test_name}"] = (test_passed, failure_reason) # Use 'sr'

            # --- MINIMAL PROGRESS UPDATE --- 
            progress_increment = (1.0 / num_tests_in_file)
            current_progress = min(initial_file_progress + (i + 1) * progress_increment, initial_file_progress + 1)
            pbar.n = max(current_progress, pbar.n)
            
            # Update description with running totals
            pass_rate = (total_passed[0] / total_tests * 100) if total_tests > 0 else 0
            pbar.set_description_str(f"Files ({total_passed[0]}✓/{total_failed[0]}✗/{total_tests} tests, {pass_rate:.1f}%)")
            pbar.refresh()

        # --- End: Modified Inner Loop ---

        # Ensure the progress bar cleanly reaches the next integer after file completion
        pbar.n = initial_file_progress + 1
        pbar.refresh()

def _run_tests_parallel(test_files, pbar, test_llm, sr, total_passed, total_failed, parallel_workers):
    """Parallel test execution using multiprocessing"""
    # Collect all test instances
    all_test_data = []
    
    for filename in test_files:
        module_name = filename[:-3]
        file_path = os.path.join("tests", filename)
        
        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if not spec or not spec.loader:
                pbar.write(f"Warning: Could not create spec for {filename}. Skipping.")
                continue
                
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            test_names_in_file = [name for name in dir(module) if name.startswith("Test") and hasattr(getattr(module, name), '__call__')]
            
            for test_name in test_names_in_file:
                # Pass file path and class name instead of instances for multiprocessing compatibility
                eval_llm_name = llm.eval_llm.original_name if llm.eval_llm else None
                vision_eval_llm_name = llm.vision_eval_llm.original_name if llm.vision_eval_llm else None
                test_data = (f"{filename}.{test_name}", file_path, test_name, test_llm.original_name, eval_llm_name, vision_eval_llm_name)
                all_test_data.append(test_data)
                
        except Exception as import_exc:
            pbar.write(f"Warning: Skipping {filename} due to import error: {import_exc}")
            continue
    
    if not all_test_data:
        pbar.write("No tests to run")
        return
    
    # Update progress bar total to reflect actual number of tests
    pbar.total = len(all_test_data)
    pbar.set_description_str(f"Running {len(all_test_data)} tests with {parallel_workers} workers")
    pbar.refresh()
    
    # Run tests in parallel with real-time progress updates
    with mp.Pool(parallel_workers) as pool:
        # Submit all jobs asynchronously
        async_results = [pool.apply_async(run_test_with_name, (test_data,)) for test_data in all_test_data]
        
        # Process results as they complete
        for async_result in async_results:
            test_name, test_passed, failure_reason = async_result.get()  # This blocks until one result is ready
            
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

def main():
    # --- Minimal Logging Configuration --- 
    log_filename = 'debug_stream.log'
    logging.basicConfig(
        level=logging.DEBUG, # Log DEBUG level and above
        format='%(asctime)s - %(levelname)s - %(name)s - [%(funcName)s] %(message)s', # Added funcName
        handlers=[
            logging.FileHandler(log_filename, mode='a'), # Append to file, allow workers to write
            logging.StreamHandler(sys.stdout) # Log INFO+ to console (stdout)
        ]
    )
    # Set console handler level (optional)
    handlers = logging.getLogger().handlers
    if len(handlers) > 1:
        handlers[1].setLevel(logging.INFO)
    logging.info(f"Logging initialized. DEBUG logs going to {log_filename}")
    # --- End Logging Configuration ---

    parser = argparse.ArgumentParser(description="Run tests on language models.")
    parser.add_argument('--model', help='Specify a specific model to run.', type=str, action="append")
    parser.add_argument('--all-models', help='Run all models.', action='store_true')
    
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
    
    if args.all_models and args.model:
        parser.error("The arguments --all-models and --model cannot be used together.")
    
    # Create the results directory if it doesn't exist
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    models_to_run = []
    if args.model:
        models_to_run = args.model
    elif args.all_models:
        models_to_run = ["gpt-4o", "gpt-4-0125-preview", "claude-3-opus-20240229", "claude-3-sonnet-20240229", "gpt-3.5-turbo-0125", "gemini-pro", "mistral-large-latest", "mistral-medium"]

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
                result = run_all_tests(model, use_cache=False,
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
            raise "Unreachable"

    if args.generate_report:
        tags, descriptions = get_tags()  # Assuming these functions are defined in your codebase
        create_results_html.generate_report(data, tags, descriptions)

if __name__ == "__main__":
    main()
