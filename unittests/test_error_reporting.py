#!/usr/bin/env python3
"""Test script to verify enhanced error reporting in evaluators"""

import sys
import os
import unittest

from evaluator import SubstringEvaluator, RegexEvaluator, EqualEvaluator
from main import format_failure_reason

class TestErrorReporting(unittest.TestCase):
    """Test that evaluators provide clear error messages with expected vs actual values"""

    def test_substring_evaluator_shows_actual_output(self):
        """Test that SubstringEvaluator includes actual output in error messages"""
        evaluator = SubstringEvaluator("expected_string")
        actual_output = "This is the actual output that does not contain what we want"
        
        for result, reason in evaluator(actual_output):
            self.assertFalse(result)  # Should fail
            error_msg = format_failure_reason(reason)
            
            # Should show both expected and actual
            self.assertIn("expected_string", error_msg)
            self.assertIn("actual output", error_msg)
            break

    def test_regex_evaluator_shows_actual_output(self):
        """Test that RegexEvaluator includes actual output in error messages"""
        evaluator = RegexEvaluator(r"\d{4}-\d{2}-\d{2}")  # Date pattern
        actual_output = "Today is Monday and it's a nice day"
        
        for result, reason in evaluator(actual_output):
            self.assertFalse(result)  # Should fail
            error_msg = format_failure_reason(reason)
            
            # Should show pattern and actual output
            self.assertTrue("RegexEvaluator" in error_msg)
            self.assertIn("Monday", error_msg)
            break

    def test_equal_evaluator_shows_actual_output(self):
        """Test that EqualEvaluator includes actual output in error messages"""
        evaluator = EqualEvaluator("expected_value")
        actual_output = "different_value"
        
        for result, reason in evaluator(actual_output):
            self.assertFalse(result)  # Should fail
            error_msg = format_failure_reason(reason)
            
            # Should show both expected and actual
            self.assertIn("expected_value", error_msg)
            self.assertIn("different_value", error_msg)
            break

if __name__ == "__main__":
    unittest.main()