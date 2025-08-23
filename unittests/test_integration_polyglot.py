#!/usr/bin/env python3
"""
Integration tests for polyglot evaluation - ONLY run when explicitly requested.

These tests use Docker and are slow. They verify the complete pipeline works.
Run with: python -m pytest unittests/test_integration_polyglot.py -m integration
"""

import pytest

# Use proper imports - pytest handles path resolution
from evaluator import PythonRun, RustRun, Env
import docker_controller

@pytest.mark.integration
def test_rust_run_execution_routing():
    """Integration test: Verify RustRun actually uses Rust compiler."""
    
    # Simple test to verify RustRun works
    rust_code = 'fn main() { println!("hello world"); }'
    
    env = Env()
    docker_controller.setup_docker(env)
    
    rust_runner = RustRun()
    rust_runner.setup(env, None, None, None, None)
    
    results = list(rust_runner(rust_code))
    output, reason = results[0]
    
    assert "hello world" in output, f"Rust execution should work: {output}"
    assert "SyntaxError" not in output, f"Should not have Python errors: {output}"

@pytest.mark.integration  
def test_polyglot_code_execution_reality():
    """Integration test: Verify polyglot attempts behave as expected."""
    
    # The problematic polyglot code from evaluations
    polyglot_code = '''/*
print("hello world")
*/
fn main() {
    println!("hello world");
}
/*
*/'''
    
    env = Env()
    docker_controller.setup_docker(env)
    
    # Test Python execution
    python_runner = PythonRun()
    python_runner.setup(env, None, None, None, None)
    python_results = list(python_runner(polyglot_code))
    python_output = python_results[0][0]
    
    # Test Rust execution  
    rust_runner = RustRun()
    rust_runner.setup(env, None, None, None, None)
    rust_results = list(rust_runner(polyglot_code))
    rust_output = rust_results[0][0]
    
    # Verify expected behavior
    assert "SyntaxError" in python_output, "Python should fail on C-style comments"
    assert "hello world" in rust_output, "Rust should succeed"
    
    print("✅ Integration test confirms: Framework working correctly")
    print("✅ Python fails (correct), Rust succeeds (correct)")

if __name__ == "__main__":
    # Only run integration tests if explicitly requested
    if "--integration" in sys.argv:
        test_rust_run_execution_routing()
        test_polyglot_code_execution_reality()
        print("\n✅ Integration tests passed!")
    else:
        print("Integration tests skipped. Use --integration flag to run.")
        print("These tests are slow because they use Docker.")