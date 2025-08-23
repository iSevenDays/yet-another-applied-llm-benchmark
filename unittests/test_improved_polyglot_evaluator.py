#!/usr/bin/env python3
"""
Test the improved PolyglotAwareEvaluator to ensure it properly evaluates
sophisticated polyglot attempts and awards appropriate credit.
"""

import pytest

# Import the improved classes using proper imports
from tests.print_hello_poly import PolyglotAwareEvaluator, CollectPolyglotResults
from evaluator import PythonRun, RustRun, Env, Reason
import docker_controller

def test_polyglot_evaluator_perfect_success():
    """Test evaluator with a perfect polyglot (if it existed)."""
    
    # Simulated perfect results
    python_results = [("hello world", None)]
    rust_results = [("hello world", None)]
    collected_results = (python_results, rust_results)
    
    evaluator = PolyglotAwareEvaluator("hello world")
    
    results = list(evaluator(collected_results))
    success, reason = results[0]
    
    assert success == True, "Perfect polyglot should succeed"
    assert "Perfect polyglot achieved" in reason.children[0], f"Wrong reason: {reason.children}"
    
    print("✓ test_polyglot_evaluator_perfect_success PASSED")

def test_polyglot_evaluator_sophisticated_failure():
    """Test evaluator recognizes sophisticated polyglot attempts where both languages fail."""
    
    # Simulated results where BOTH languages fail due to polyglot attempt
    python_error = '  File "/usr/src/app/main.py", line 1\n    /*\n    ^\nSyntaxError: invalid syntax\n'
    python_reason = (None, ('/*\nprint("hello world")\n*/\nfn main() {\n    println!("hello world");\n}\n', python_error))
    python_results = [(python_error, python_reason)]
    
    # Rust also fails for some reason (to test the C-style comment recognition)
    rust_results = [("compilation error", None)]
    
    collected_results = (python_results, rust_results)
    
    evaluator = PolyglotAwareEvaluator("hello world")
    
    results = list(evaluator(collected_results))
    success, reason = results[0]
    
    # Should fail but recognize the sophisticated attempt
    assert success == False, "Sophisticated attempt should still fail overall"
    assert "C-style comments" in reason.children[0], f"Should recognize polyglot technique: {reason.children}"
    
    print("✓ test_polyglot_evaluator_sophisticated_failure PASSED")

def test_polyglot_evaluator_partial_success_recognizes_attempt():
    """Test evaluator recognizes sophisticated polyglot attempts even with partial success."""
    
    # This is actually what happens with Qwen3-235B - Python fails, Rust succeeds
    python_error = '  File "/usr/src/app/main.py", line 1\n    /*\n    ^\nSyntaxError: invalid syntax\n'
    python_reason = (None, ('/*\nprint("hello world")\n*/\nfn main() {\n    println!("hello world");\n}\n', python_error))
    python_results = [(python_error, python_reason)]
    
    rust_results = [("hello world\n", None)]
    
    collected_results = (python_results, rust_results)
    
    evaluator = PolyglotAwareEvaluator("hello world")
    
    results = list(evaluator(collected_results))
    success, reason = results[0]
    
    # Should fail but recognize both partial success AND sophisticated attempt
    assert success == False, "Partial success should still fail overall test"
    assert ("Partial success: Rust works" in reason.children[0] or 
            "polyglot understanding demonstrated" in reason.children[0]), f"Should recognize Rust success and polyglot attempt: {reason.children}"
    
    print("✓ test_polyglot_evaluator_partial_success_recognizes_attempt PASSED")

def test_polyglot_evaluator_partial_success():
    """Test evaluator recognizes partial success (one language works)."""
    
    python_results = [("", None)]  # Python failed
    rust_results = [("hello world", None)]  # Rust worked
    
    collected_results = (python_results, rust_results)
    
    evaluator = PolyglotAwareEvaluator("hello world")
    
    results = list(evaluator(collected_results))
    success, reason = results[0]
    
    assert success == False, "Partial success should still fail overall test"
    assert "Partial success: Rust works" in reason.children[0], f"Should recognize Rust success: {reason.children}"
    
    print("✓ test_polyglot_evaluator_partial_success PASSED")

def test_collect_polyglot_results():
    """Test the CollectPolyglotResults helper node."""
    
    # Setup environment
    env = Env()
    docker_controller.setup_docker(env)
    
    # Test with the problematic polyglot code
    test_code = '''/*
print("hello world")
*/
fn main() {
    println!("hello world");
}
/*
*/'''
    
    collector = CollectPolyglotResults(PythonRun(), RustRun())
    collector.setup(env, None, None, None, None)
    
    results = list(collector(test_code))
    collected_results, reason = results[0]
    
    python_results, rust_results = collected_results
    
    # Verify structure
    assert len(python_results) == 1, "Should have one Python result"
    assert len(rust_results) == 1, "Should have one Rust result"
    
    # Verify Python fails due to syntax
    python_output = python_results[0][0]
    assert "SyntaxError" in python_output, f"Python should fail with syntax error: {python_output}"
    
    # Verify Rust succeeds
    rust_output = rust_results[0][0]
    assert "hello world" in rust_output, f"Rust should succeed: {rust_output}"
    
    print("✓ test_collect_polyglot_results PASSED")

def test_complete_improved_pipeline():
    """Test the complete improved evaluation pipeline."""
    
    # Setup environment
    env = Env()
    docker_controller.setup_docker(env)
    
    # Test with sophisticated polyglot attempt
    test_code = '''/*
print("hello world")
*/
fn main() {
    println!("hello world");
}
/*
*/'''
    
    # Run complete pipeline
    collector = CollectPolyglotResults(PythonRun(), RustRun())
    collector.setup(env, None, None, None, None)
    
    evaluator = PolyglotAwareEvaluator("hello world")
    
    # Execute pipeline
    for collected_results, collect_reason in collector(test_code):
        for final_result, eval_reason in evaluator(collected_results):
            
            print(f"Final result: {final_result}")
            print(f"Evaluation reason: {eval_reason.children}")
            
            # Should recognize sophisticated attempt even though it fails
            assert ("polyglot" in eval_reason.children[0].lower() or 
                   "C-style comments" in eval_reason.children[0] or
                   "partial success" in eval_reason.children[0].lower()), f"Should recognize polyglot attempt: {eval_reason.children[0]}"
            
            print("✓ test_complete_improved_pipeline PASSED")
            return

if __name__ == "__main__":
    print("Testing improved PolyglotAwareEvaluator...")
    
    try:
        test_polyglot_evaluator_perfect_success()
        test_polyglot_evaluator_sophisticated_failure()
        test_polyglot_evaluator_partial_success_recognizes_attempt()
        test_polyglot_evaluator_partial_success()
        test_collect_polyglot_results()
        test_complete_improved_pipeline()
        
        print("\n✅ ALL TESTS PASSED - Improved evaluator working correctly!")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)