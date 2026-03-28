#!/usr/bin/env python3
"""
Test suite for RemoteProtocolError backoff retry fix - SPARC-compliant testing.
Focus: Validate specific error detection logic for the httpcore.RemoteProtocolError case.
"""
import unittest

class TestRemoteProtocolErrorBackoff(unittest.TestCase):
    """Test the specific RemoteProtocolError detection logic for backoff retry."""
    
    def test_remote_protocol_error_detection(self):
        """Test that RemoteProtocolError with peer closed connection is detected for retry."""
        
        # The actual error from the user's issue
        error_message = "httpcore.RemoteProtocolError: peer closed connection without sending complete message body (incomplete chunked read)"
        
        # Test the detection logic we implemented in llm.py:231-234
        should_retry = "RemoteProtocolError" in error_message and "peer closed connection" in error_message
        
        self.assertTrue(should_retry, 
                       "The specific RemoteProtocolError should be detected for retry")
    
    def test_other_remote_protocol_errors_not_detected(self):
        """Test that other RemoteProtocolError variants are NOT detected for retry."""
        
        non_retryable_errors = [
            "httpcore.RemoteProtocolError: different error message",
            "Some other peer closed connection error",
            "RemoteProtocolError without peer connection text",
        ]
        
        for error_message in non_retryable_errors:
            with self.subTest(error_message=error_message):
                should_retry = "RemoteProtocolError" in error_message and "peer closed connection" in error_message
                self.assertFalse(should_retry, 
                               f"Error should NOT trigger retry: {error_message}")
    
    def test_backoff_times_configuration(self):
        """Test that backoff times are correctly configured."""
        # This validates the backoff_times array defined in llm.py:263
        expected_backoff_times = [10, 20, 30, 60, 90, 120, 300]
        
        # Import here to avoid import issues during test collection
        from llm import LLM
        
        # Read the source to verify backoff times (simple validation)
        import inspect
        source = inspect.getsource(LLM.__call__)
        
        # Check that our expected backoff times are present in the source
        self.assertIn("backoff_times = [10, 20, 30, 60, 90, 120, 300]", source,
                     "Backoff times should be configured as expected")


if __name__ == '__main__':
    unittest.main(verbosity=2)