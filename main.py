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

from evaluator import Env, Conversation, run_test

import multiprocessing as mp

def run_one_test(test, test_llm, eval_llm, vision_eval_llm):
    """
    Runs just one test case and returns either true or false and the output.
    """
    import docker_controller
    env = Env()
    test.setup(env, Conversation(test_llm), test_llm, eval_llm, vision_eval_llm)

    try:
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
    except Exception as e:
        # Handle any other exceptions that might occur
        if env.container:
            docker_controller.async_kill_container(env.docker, env.container)
        return False, f"An error occurred: {str(e)}"
                    

def run_all_tests(test_llm_name, use_cache=True, which_tests=None):
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
    
    # Create a tqdm progress bar
    with tqdm(total=len(test_files), desc="Files", unit="file", ascii=" >==") as pbar:
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
                pbar.set_description_str(f"Test: {filename} -> {test_name}") # Update description
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

                result_str = "PASS" if test_passed else "FAIL"
                pbar.write(f"{result_str}: {test_name}")
                if not test_passed:
                    reason_str = repr(failure_reason)
                    if len(reason_str) > 300: reason_str = reason_str[:297] + "..."
                    pbar.write(f"  Reason: {reason_str}")

                sr[f"{filename}.{test_name}"] = (test_passed, failure_reason) # Use 'sr'

                # --- MINIMAL PROGRESS UPDATE --- 
                progress_increment = (1.0 / num_tests_in_file)
                current_progress = min(initial_file_progress + (i + 1) * progress_increment, initial_file_progress + 1)
                pbar.n = max(current_progress, pbar.n)
                pbar.refresh()

            # --- End: Modified Inner Loop ---

            # Ensure the progress bar cleanly reaches the next integer after file completion
            pbar.n = initial_file_progress + 1
            pbar.refresh()
    
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
        except:
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
            logging.FileHandler(log_filename, mode='w'), # Write to file, overwrite each run
            logging.StreamHandler(sys.stdout) # Log INFO+ to console (stdout)
        ]
    )
    # Set console handler level (optional)
    logging.getLogger().handlers[1].setLevel(logging.INFO)
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
                                       which_tests=tests_subset)

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
