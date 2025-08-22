#!/usr/bin/env python3
"""
Simple Python wrapper for the Go thinking handler shared library.
Go-only implementation with no fallbacks.
"""

import ctypes
import os
import platform
from typing import Optional


class GoThinkingHandler:
    """Python wrapper for the Go thinking handler shared library."""
    
    def __init__(self):
        self._lib: Optional[ctypes.CDLL] = None
        self._load_library()
        if not self._lib:
            raise RuntimeError("Go thinking handler library could not be loaded")
        
    def _load_library(self):
        """Load the Go shared library."""
        # Look for the library in the go/ subdirectory
        go_dir = os.path.join(os.path.dirname(__file__), "go")
        lib_path = os.path.join(go_dir, "libthinking_handler.so")
        
        if not os.path.exists(lib_path):
            print(f"Go library not found at {lib_path}")
            return
            
        try:
            self._lib = ctypes.CDLL(lib_path)
            self._setup_function_signatures()
            print("Go thinking handler library loaded successfully")
        except Exception as e:
            print(f"Failed to load Go library: {e}")
            self._lib = None
    
    def _setup_function_signatures(self):
        """Setup ctypes function signatures for the Go library."""
        if not self._lib:
            return
            
        # has_thinking_tokens(text *C.char) C.int
        self._lib.has_thinking_tokens.argtypes = [ctypes.c_char_p]
        self._lib.has_thinking_tokens.restype = ctypes.c_int
        
        # strip_thinking_tokens(text *C.char) *C.char
        self._lib.strip_thinking_tokens.argtypes = [ctypes.c_char_p]
        self._lib.strip_thinking_tokens.restype = ctypes.c_char_p
        
        # convert_thinking_to_collapsible_html(text *C.char) *C.char
        self._lib.convert_thinking_to_collapsible_html.argtypes = [ctypes.c_char_p]
        self._lib.convert_thinking_to_collapsible_html.restype = ctypes.c_char_p
        
        # get_css_styles() *C.char
        self._lib.get_css_styles.argtypes = []
        self._lib.get_css_styles.restype = ctypes.c_char_p
        
        # get_javascript() *C.char
        self._lib.get_javascript.argtypes = []
        self._lib.get_javascript.restype = ctypes.c_char_p
    
    def has_thinking_tokens(self, text: str) -> bool:
        """Check if text contains any thinking tokens."""
        text_bytes = text.encode('utf-8')
        result = self._lib.has_thinking_tokens(ctypes.c_char_p(text_bytes))
        return result == 1
    
    def strip_thinking_tokens(self, text: str) -> str:
        """Remove all thinking tokens from text."""
        text_bytes = text.encode('utf-8')
        result_ptr = self._lib.strip_thinking_tokens(ctypes.c_char_p(text_bytes))
        if result_ptr:
            result = ctypes.string_at(result_ptr).decode('utf-8')
            return result
        return text
    
    def convert_to_collapsible_html(self, text: str) -> str:
        """Convert thinking tokens to collapsible HTML elements."""
        text_bytes = text.encode('utf-8')
        result_ptr = self._lib.convert_thinking_to_collapsible_html(ctypes.c_char_p(text_bytes))
        if result_ptr:
            result = ctypes.string_at(result_ptr).decode('utf-8')
            return result
        return text
    
    def get_css_styles(self) -> str:
        """Get CSS styles for thinking containers."""
        result_ptr = self._lib.get_css_styles()
        if result_ptr:
            result = ctypes.string_at(result_ptr).decode('utf-8')
            return result
        return ""
    
    def get_javascript(self) -> str:
        """Get JavaScript for toggle functionality."""
        result_ptr = self._lib.get_javascript()
        if result_ptr:
            result = ctypes.string_at(result_ptr).decode('utf-8')
            return result
        return ""


# Global instance
go_thinking_handler = GoThinkingHandler()

# Convenience functions
def strip_thinking_tokens(text: str) -> str:
    """Remove thinking tokens from text."""
    return go_thinking_handler.strip_thinking_tokens(text)

def has_thinking_tokens(text: str) -> bool:
    """Check if text has thinking tokens."""
    return go_thinking_handler.has_thinking_tokens(text)

def convert_thinking_to_collapsible_html(text: str) -> str:
    """Convert thinking to HTML."""
    return go_thinking_handler.convert_to_collapsible_html(text)


if __name__ == "__main__":
    # Simple test
    test_text = "Hello <think>Let me think about this</think> World!"
    print(f"Has thinking tokens: {has_thinking_tokens(test_text)}")
    print(f"Stripped: {strip_thinking_tokens(test_text)}")
    print("Testing complete!")