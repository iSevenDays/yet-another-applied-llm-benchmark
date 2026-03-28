"""
Test-driven development for enhanced Reason.describe_failure() method.
Following SPARC TDD principles: Simple, focused tests for each node type.
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluator import Reason


class MockNode:
    """Mock node with configurable name."""
    def __init__(self, name):
        self.__name__ = name


class TestReasonDescribeFailure(unittest.TestCase):
    """Test the describe_failure method on Reason objects."""
    
    def test_python_run_execution_error(self):
        """Should describe Python execution errors clearly."""
        node = MockNode('PythonRun')
        code = 'print(undefined_variable)'
        error_output = 'NameError: name "undefined_variable" is not defined'
        reason = Reason(node, [code, error_output])
        
        description = reason.describe_failure()
        
        self.assertIn('PythonRun', description)
        self.assertIn('NameError', description)
        self.assertIn('undefined_variable', description)
    
    def test_substring_evaluator_failure(self):
        """Should describe evaluator failures with expected vs actual."""
        node = MockNode('SubstringEvaluator')
        expected = 'success'
        result = False
        actual = 'failure occurred'
        reason = Reason(node, [expected, result, actual])
        
        description = reason.describe_failure()
        
        self.assertIn('SubstringEvaluator', description)
        self.assertIn('success', description)
        self.assertIn('failure occurred', description)
    
    def test_then_node_pipeline_failure(self):
        """Should describe pipeline failures with step context."""
        python_node = MockNode('PythonRun')
        python_reason = Reason(python_node, ['code', 'SyntaxError: bad syntax'])
        
        eval_node = MockNode('SubstringEvaluator')
        eval_reason = Reason(eval_node, ['expected', True, 'output'])
        
        then_node = MockNode('ThenNode')
        then_reason = Reason(then_node, [python_reason, eval_reason])
        
        description = then_reason.describe_failure()
        
        self.assertIn('ThenNode', description)
        self.assertIn('PythonRun', description)
        self.assertIn('SubstringEvaluator', description)
        self.assertIn('SyntaxError', description)
    
    def test_bash_run_command_error(self):
        """Should describe Bash execution errors."""
        node = MockNode('BashRun')
        command = 'nonexistent_command'
        error_output = 'bash: nonexistent_command: command not found'
        reason = Reason(node, [command, error_output])
        
        description = reason.describe_failure()
        
        self.assertIn('BashRun', description)
        self.assertIn('command not found', description)
    
    def test_timeout_error(self):
        """Should describe timeout errors clearly."""
        node = MockNode('PythonRun')
        code = 'while True: pass'
        timeout_output = 'Timeout: function took too long to complete'
        reason = Reason(node, [code, timeout_output])
        
        description = reason.describe_failure()
        
        self.assertIn('PythonRun', description)
        self.assertIn('Timeout', description)
    
    def test_unknown_node_fallback(self):
        """Should provide reasonable fallback for unknown node types."""
        node = MockNode('UnknownNode')
        reason = Reason(node, ['some', 'data'])
        
        description = reason.describe_failure()
        
        self.assertIn('UnknownNode', description)
        self.assertIn('failed', description.lower())
    
    def test_max_length_respected(self):
        """Should respect maximum length constraints."""
        node = MockNode('PythonRun')
        very_long_error = 'x' * 1000
        reason = Reason(node, ['code', very_long_error])
        
        description = reason.describe_failure(max_length=100)
        
        self.assertLessEqual(len(description), 100)
        self.assertIn('...', description)
    
    def test_successful_evaluator_no_failure_description(self):
        """Should return None or minimal description for successful evaluators."""
        node = MockNode('SubstringEvaluator')
        reason = Reason(node, ['expected', True, 'expected output found'])
        
        description = reason.describe_failure()
        
        # Should either be None or a very brief success message
        if description is not None:
            self.assertLess(len(description), 50)


class TestReasonDescribeFailureIntegration(unittest.TestCase):
    """Integration tests with real node types (if available)."""
    
    def test_nested_pipeline_error_description(self):
        """Should handle deeply nested pipeline failures."""
        # Inner execution error
        inner_node = MockNode('PythonRun')
        inner_reason = Reason(inner_node, ['bad_code', 'IndentationError: bad indent'])
        
        # Middle pipeline
        middle_node = MockNode('AndNode')  
        middle_reason = Reason(middle_node, [inner_reason, 'other'])
        
        # Outer pipeline
        outer_node = MockNode('ThenNode')
        outer_reason = Reason(outer_node, [middle_reason, 'final'])
        
        description = outer_reason.describe_failure()
        
        self.assertIn('ThenNode', description)
        self.assertIn('IndentationError', description)
    
    def test_multiple_error_extraction(self):
        """Should extract multiple error lines from output."""
        node = MockNode('PythonRun')
        multi_error = """
        Traceback (most recent call last):
          File "main.py", line 1
            bad syntax here
                     ^
        SyntaxError: invalid syntax
        ModuleNotFoundError: No module named 'missing'
        """
        reason = Reason(node, ['code', multi_error])
        
        description = reason.describe_failure()
        
        self.assertIn('SyntaxError', description)
        # Should capture key error information


if __name__ == '__main__':
    unittest.main()