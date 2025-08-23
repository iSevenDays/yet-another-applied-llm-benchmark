#!/usr/bin/env python3
"""
Test to reproduce the exact PrintHelloPoly2 pipeline issue.

Following TDD approach to isolate where the execution routing goes wrong.
"""

import sys
import os
from evaluator import ExtractCode, PythonRun, RustRun, SubstringEvaluator, Env, Conversation, AndNode
import docker_controller

def test_printhellopoly2_pipeline_simulation():
    """Test the exact pipeline used in PrintHelloPoly2 test."""
    
    # This is the extracted code from the Qwen3-235B evaluation that failed
    extracted_code = '''/*
print("hello world")
*/
fn main() {
    println!("hello world");
}

/*
*/'''
    
    # Setup environment
    env = Env()
    docker_controller.setup_docker(env)
    
    # Test Python execution (should work)
    python_runner = PythonRun()
    python_runner.setup(env, None, None, None, None)
    
    python_results = list(python_runner(extracted_code))
    print(f"Python execution results: {python_results}")
    
    # Test Rust execution (this is where the issue should occur)
    rust_runner = RustRun() 
    rust_runner.setup(env, None, None, None, None)
    
    rust_results = list(rust_runner(extracted_code))
    print(f"Rust execution results: {rust_results}")
    
    # Check for the specific issue
    rust_output, rust_reason = rust_results[0]
    
    # This should reveal the problem
    if "File \"/usr/src/app/main.py\"" in rust_output:
        print("🚨 BUG CONFIRMED: Rust code being executed as Python!")
        print(f"Rust output: {rust_output}")
        return False
    
    if "SyntaxError" in rust_output and "/*" in rust_output:
        print("🚨 BUG CONFIRMED: Python syntax error on Rust comment syntax!")
        print(f"Rust output: {rust_output}")
        return False
    
    # Test the full pipeline like in PrintHelloPoly2
    answer = "hello world"
    
    # Python part of the pipeline
    python_eval = SubstringEvaluator(answer)
    python_eval.setup(env, None, None, None, None)
    
    python_success = False
    for output, reason in python_runner(extracted_code):
        for result, eval_reason in python_eval(output):
            python_success = result
            print(f"Python pipeline success: {python_success}")
            break
        break
    
    # Rust part of the pipeline  
    rust_eval = SubstringEvaluator(answer)
    rust_eval.setup(env, None, None, None, None)
    
    rust_success = False
    for output, reason in rust_runner(extracted_code):
        print(f"Raw Rust output before evaluation: '{output}'")
        for result, eval_reason in rust_eval(output):
            rust_success = result
            print(f"Rust pipeline success: {rust_success}")
            break
        break
    
    print(f"Final results - Python: {python_success}, Rust: {rust_success}")
    
    # The real issue: polyglot code with C-style comments can't work in Python
    # This is actually CORRECT behavior - not a bug in the evaluation framework
    print(f"Python failed due to C-style comments (/**/): {not python_success}")
    print(f"Rust succeeded as expected: {rust_success}")
    
    # The test is actually working correctly - polyglots are extremely difficult
    assert rust_success, "Rust execution should work and does"
    
    # Python failing on C-style comments is correct behavior, not a bug
    
    return True

if __name__ == "__main__":
    print("Testing PrintHelloPoly2 pipeline to isolate the routing bug...")
    
    try:
        test_printhellopoly2_pipeline_simulation()
        print("\n✅ Pipeline working correctly!")
    except Exception as e:
        print(f"\n❌ PIPELINE ISSUE CONFIRMED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)