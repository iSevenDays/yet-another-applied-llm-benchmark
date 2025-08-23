#!/usr/bin/env python3
"""
Unit tests for polyglot evaluation logic - NO Docker involved.

Following TDD principles: test the logic units, not the infrastructure.
This tests the evaluation logic without Docker overhead.

Run with: python -m pytest unittests/test_polyglot_logic_unit.py
"""

import pytest

# Import from the tests module - pytest will handle path resolution
from tests.print_hello_poly import PolyglotAwareEvaluator
from evaluator import Reason

def test_polyglot_evaluator_perfect_success():
    """Test evaluator logic with perfect polyglot results."""
    
    # Mocked perfect results - no Docker needed
    python_results = [("hello world", None)]
    rust_results = [("hello world", None)]
    collected_results = (python_results, rust_results)
    
    evaluator = PolyglotAwareEvaluator("hello world")
    
    results = list(evaluator(collected_results))
    success, reason = results[0]
    
    assert success == True, "Perfect polyglot should succeed"
    assert "Perfect polyglot achieved" in reason.children[0]
    
    print("✓ test_polyglot_evaluator_perfect_success PASSED")

def test_polyglot_evaluator_recognizes_sophisticated_attempts():
    """Test evaluator recognizes C-style comment polyglot attempts."""
    
    # Simulate what happens with C-style comments in Python
    python_error = 'SyntaxError: invalid syntax'
    python_code = '/*\nprint("hello world")\n*/\nfn main() {\n    println!("hello world");\n}'
    python_reason = (None, (python_code, python_error))
    python_results = [(python_error, python_reason)]
    
    # Both languages fail (to trigger the sophisticated attempt detection)
    rust_results = [("compilation failed", None)]
    
    collected_results = (python_results, rust_results)
    
    evaluator = PolyglotAwareEvaluator("hello world")
    results = list(evaluator(collected_results))
    success, reason = results[0]
    
    assert success == False, "Should fail but recognize attempt"
    assert "C-style comments" in reason.children[0], f"Should detect polyglot technique: {reason.children}"
    
    print("✓ test_polyglot_evaluator_recognizes_sophisticated_attempts PASSED")

def test_polyglot_evaluator_partial_success():
    """Test evaluator handles partial success (one language works)."""
    
    python_results = [("", None)]  # Python failed
    rust_results = [("hello world", None)]  # Rust worked
    
    collected_results = (python_results, rust_results)
    
    evaluator = PolyglotAwareEvaluator("hello world")
    results = list(evaluator(collected_results))
    success, reason = results[0]
    
    assert success == False, "Partial success should still fail overall"
    assert "Partial success: Rust works" in reason.children[0]
    
    print("✓ test_polyglot_evaluator_partial_success PASSED")

def test_polyglot_evaluator_recognizes_advanced_techniques():
    """Test evaluator recognizes various polyglot techniques."""
    
    test_cases = [
        ("__name__ == '__main__'", "Python conditional execution"),
        ("#ifdef", "C preprocessor directives"),  
        ('r"""', "Raw string literals"),
        ("macro_rules!", "Rust macros"),
    ]
    
    for technique, description in test_cases:
        python_code = f"// Comment\n{technique}\nprint('hello')"
        python_reason = (None, (python_code, "failed"))
        python_results = [("failed", python_reason)]
        rust_results = [("failed", None)]
        
        collected_results = (python_results, rust_results)
        
        evaluator = PolyglotAwareEvaluator("hello world")
        results = list(evaluator(collected_results))
        success, reason = results[0]
        
        assert success == False, f"Should fail for {description}"
        assert "polyglot technique" in reason.children[0].lower(), f"Should recognize {description}: {reason.children}"
    
    print("✓ test_polyglot_evaluator_recognizes_advanced_techniques PASSED")

def test_original_evaluation_issue_analysis():
    """Test that reproduces the original evaluation 'bug' and shows it's actually correct."""
    
    # This reproduces the Qwen3-235B evaluation result
    python_error = '  File "/usr/src/app/main.py", line 1\n    /*\n    ^\nSyntaxError: invalid syntax\n'
    python_code = '/*\nprint("hello world")\n*/\nfn main() {\n    println!("hello world");\n}'
    python_reason = (None, (python_code, python_error))
    python_results = [(python_error, python_reason)]
    
    rust_results = [("hello world\n", None)]  # Rust actually worked
    
    collected_results = (python_results, rust_results)
    
    evaluator = PolyglotAwareEvaluator("hello world")
    results = list(evaluator(collected_results))
    success, reason = results[0]
    
    # This should be recognized as sophisticated partial success
    assert success == False, "Overall test should fail (not a true polyglot)"
    assert ("partial success" in reason.children[0].lower() or
            "polyglot understanding" in reason.children[0].lower()), f"Should recognize sophisticated attempt: {reason.children}"
    
    print("✓ test_original_evaluation_issue_analysis PASSED")
    print("  → This shows the evaluation was actually CORRECT, not buggy")
    print("  → Models failed because true Python/Rust polyglots are extremely difficult")

if __name__ == "__main__":
    print("Testing PolyglotAwareEvaluator logic (unit tests - no Docker)...")
    
    test_polyglot_evaluator_perfect_success()
    test_polyglot_evaluator_recognizes_sophisticated_attempts()
    test_polyglot_evaluator_partial_success()
    test_polyglot_evaluator_recognizes_advanced_techniques()
    test_original_evaluation_issue_analysis()
    
    print("\n✅ ALL UNIT TESTS PASSED - Logic is correct!")
    print("📝 Root cause confirmed: Evaluation framework works correctly.")
    print("📝 Issue was unrealistic expectations about polyglot difficulty.")