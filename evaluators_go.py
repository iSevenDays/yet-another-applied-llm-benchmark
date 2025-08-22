#!/usr/bin/env python3
"""
Python wrapper for the Go evaluators shared library.
Provides high-performance regex, substring, and integer evaluation.
"""

import ctypes
import os
from typing import Optional


class GoEvaluators:
    """Python wrapper for the Go evaluators shared library."""
    
    def __init__(self):
        self._lib: Optional[ctypes.CDLL] = None
        self._load_library()
        if not self._lib:
            raise RuntimeError("Go evaluators library could not be loaded")
        
    def _load_library(self):
        """Load the Go shared library."""
        # Look for the library in the go/ subdirectory
        go_dir = os.path.join(os.path.dirname(__file__), "go")
        lib_path = os.path.join(go_dir, "libevaluators.so")
        
        if not os.path.exists(lib_path):
            print(f"Go evaluators library not found at {lib_path}")
            return
            
        try:
            self._lib = ctypes.CDLL(lib_path)
            self._setup_function_signatures()
            print("Go evaluators library loaded successfully")
        except Exception as e:
            print(f"Failed to load Go evaluators library: {e}")
            self._lib = None
    
    def _setup_function_signatures(self):
        """Setup ctypes function signatures for the Go library."""
        if not self._lib:
            return
            
        # evaluate_regex(pattern *C.char, text *C.char, ignore_case C.int) C.int
        self._lib.evaluate_regex.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        self._lib.evaluate_regex.restype = ctypes.c_int
        
        # evaluate_substring(substr *C.char, text *C.char, case_insensitive C.int) C.int
        self._lib.evaluate_substring.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        self._lib.evaluate_substring.restype = ctypes.c_int
        
        # contains_integer(number C.int, text *C.char) C.int
        self._lib.contains_integer.argtypes = [ctypes.c_int, ctypes.c_char_p]
        self._lib.contains_integer.restype = ctypes.c_int
        
        # clear_evaluator_caches()
        self._lib.clear_evaluator_caches.argtypes = []
        self._lib.clear_evaluator_caches.restype = None
    
    def evaluate_regex(self, pattern: str, text: str, ignore_case: bool = False) -> bool:
        """Evaluate if text matches the regex pattern."""
        pattern_bytes = pattern.encode('utf-8')
        text_bytes = text.encode('utf-8')
        ignore_case_flag = 1 if ignore_case else 0
        
        result = self._lib.evaluate_regex(
            ctypes.c_char_p(pattern_bytes),
            ctypes.c_char_p(text_bytes),
            ctypes.c_int(ignore_case_flag)
        )
        return result == 1
    
    def evaluate_substring(self, substring: str, text: str, case_insensitive: bool = False) -> bool:
        """Evaluate if text contains the substring."""
        substring_bytes = substring.encode('utf-8')
        text_bytes = text.encode('utf-8')
        case_insensitive_flag = 1 if case_insensitive else 0
        
        result = self._lib.evaluate_substring(
            ctypes.c_char_p(substring_bytes),
            ctypes.c_char_p(text_bytes),
            ctypes.c_int(case_insensitive_flag)
        )
        return result == 1
    
    def contains_integer(self, number: int, text: str) -> bool:
        """Evaluate if text contains the specified integer."""
        text_bytes = text.encode('utf-8')
        
        result = self._lib.contains_integer(
            ctypes.c_int(number),
            ctypes.c_char_p(text_bytes)
        )
        return result == 1
    
    def clear_caches(self):
        """Clear all internal caches in the Go library."""
        self._lib.clear_evaluator_caches()


# Global instance
go_evaluators = GoEvaluators()

# Convenience functions for backward compatibility
def evaluate_regex(pattern: str, text: str, ignore_case: bool = False) -> bool:
    """Evaluate regex pattern against text."""
    return go_evaluators.evaluate_regex(pattern, text, ignore_case)

def evaluate_substring(substring: str, text: str, case_insensitive: bool = False) -> bool:
    """Evaluate substring presence in text."""
    return go_evaluators.evaluate_substring(substring, text, case_insensitive)

def contains_integer(number: int, text: str) -> bool:
    """Check if text contains the specified integer."""
    return go_evaluators.contains_integer(number, text)


if __name__ == "__main__":
    # Simple test when run directly
    print("Testing Go evaluators...")
    
    # Test regex
    pattern = r'\d+'
    print(f"Regex test: {evaluate_regex(pattern, 'age is 25', False)}")  # Should be True
    
    # Test substring
    print(f"Substring test: {evaluate_substring('hello', 'Hello World', True)}")  # Should be True
    
    # Test integer
    print(f"Integer test: {contains_integer(42, 'The answer is 42')}")  # Should be True
    
    print("Testing complete!")