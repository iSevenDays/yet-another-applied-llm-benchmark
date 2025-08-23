#!/usr/bin/env python3
"""
Test to reproduce and verify the RustRun execution routing issue.

Following TDD approach to ensure RustRun properly executes Rust code
instead of incorrectly routing through Python interpreter.
"""

import pytest

from evaluator import RustRun, Env, Conversation  
import docker_controller

def test_rust_run_simple_hello_world():
    """Test that RustRun properly compiles and executes Rust code."""
    
    # Simple Rust hello world program
    rust_code = '''fn main() {
    println!("hello world");
}'''
    
    # Setup environment (following SPARC - Simplicity principle)
    env = Env()
    docker_controller.setup_docker(env)
    
    # Create RustRun node
    rust_runner = RustRun()
    rust_runner.setup(env, None, None, None, None)
    
    # Execute the code
    results = list(rust_runner(rust_code))
    
    # Verify results
    assert len(results) == 1, f"Expected 1 result, got {len(results)}"
    output, reason = results[0]
    
    # Critical test: output should contain "hello world" 
    assert "hello world" in output, f"Expected 'hello world' in output, got: {output}"
    
    # Critical test: should NOT contain Python syntax errors
    assert "SyntaxError" not in output, f"Should not have Python syntax errors, got: {output}"
    assert "File \"/usr/src/app/main.py\"" not in output, f"Should not reference main.py, got: {output}"
    
    print("✓ test_rust_run_simple_hello_world PASSED")
    return True

def test_rust_run_polyglot_scenario():
    """Test RustRun with polyglot-style code that would break in Python."""
    
    # Code that should work in Rust but fail if run through Python
    rust_code = '''/*
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
    
    # Create RustRun node
    rust_runner = RustRun()
    rust_runner.setup(env, None, None, None, None)
    
    # Execute the code
    results = list(rust_runner(rust_code))
    
    # Verify results
    assert len(results) == 1, f"Expected 1 result, got {len(results)}"
    output, reason = results[0]
    
    # Critical test: should successfully compile and run as Rust
    assert "hello world" in output, f"Expected 'hello world' in output, got: {output}"
    
    # Critical test: should NOT try to parse as Python
    assert "SyntaxError" not in output, f"Should not have Python syntax errors, got: {output}"
    assert "invalid syntax" not in output, f"Should not have Python parsing errors, got: {output}"
    
    print("✓ test_rust_run_polyglot_scenario PASSED")
    return True

if __name__ == "__main__":
    print("Running TDD tests for RustRun execution routing...")
    
    try:
        test_rust_run_simple_hello_world()
        test_rust_run_polyglot_scenario()
        print("\n✅ ALL TESTS PASSED - RustRun is working correctly!")
    except Exception as e:
        print(f"\n❌ TEST FAILED - Found the bug: {e}")
        print("This confirms the issue needs to be fixed.")
        sys.exit(1)
    
    sys.exit(0)