#!/usr/bin/env python3
"""
Test suite for LLM cache functionality - SPARC-compliant testing.
Focus: Clean, maintainable cache behavior validation.
"""
import unittest
import tempfile
import os
import pickle
from llm_cache import LLMCache


class TestLLMCache(unittest.TestCase):
    """Test the SPARC-compliant LLM cache implementation."""
    
    def setUp(self):
        """Set up test environment with temporary cache directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = LLMCache("test-model", cache_dir=self.temp_dir)
        
    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_cache_key_consistency(self):
        """Same inputs should always generate the same cache key."""
        conversation = ["Hello world"]
        
        key1 = self.cache.get_cache_key(conversation)
        key2 = self.cache.get_cache_key(conversation)
        
        self.assertEqual(key1, key2, "Same inputs should generate same cache key")
    
    def test_cache_key_with_parameters(self):
        """Cache keys should normalize parameters consistently for backward compatibility."""
        conversation = ["Test prompt"]
        
        # Key with explicit max_tokens only
        key_with_tokens = self.cache.get_cache_key(conversation, max_tokens=4096)
        
        # Key with hparams containing max_tokens
        key_with_hparams = self.cache.get_cache_key(
            conversation, 
            hparams={'max_tokens': 4096, 'temperature': 0.7}
        )
        
        # Key with both (explicit max_tokens takes precedence)
        key_with_both = self.cache.get_cache_key(
            conversation,
            max_tokens=8192,
            hparams={'max_tokens': 4096, 'temperature': 0.7}
        )
        
        # All should normalize max_tokens into hparams component
        self.assertIn(('hparams', (('max_tokens', 4096),)), key_with_tokens)
        self.assertIn(('hparams', (('max_tokens', 4096), ('temperature', 0.7))), key_with_hparams)
        
        # Explicit max_tokens should take precedence
        self.assertIn(('hparams', (('max_tokens', 8192), ('temperature', 0.7))), key_with_both)
    
    def test_cache_basic_operations(self):
        """Basic cache store and retrieve functionality."""
        conversation = ["Test prompt"]
        cache_key = self.cache.get_cache_key(conversation)
        response = "Test response"
        
        # Store response
        self.cache.put(cache_key, response)
        
        # Retrieve response
        cached_response = self.cache.get(cache_key)
        
        self.assertEqual(cached_response, response, "Should retrieve stored response")
    
    def test_cache_miss_returns_none(self):
        """Cache miss should return None."""
        conversation = ["Non-existent prompt"]
        cache_key = self.cache.get_cache_key(conversation)
        
        result = self.cache.get(cache_key)
        
        self.assertIsNone(result, "Cache miss should return None")
    
    def test_cache_persistence_across_instances(self):
        """Cache should persist across different cache instances."""
        conversation = ["Persistent test"]
        cache_key = self.cache.get_cache_key(conversation)
        response = "Persistent response"
        
        # Store in first instance
        self.cache.put(cache_key, response)
        
        # Create new instance (simulates new worker process)
        new_cache = LLMCache("test-model", cache_dir=self.temp_dir)
        
        # Should retrieve from new instance
        cached_response = new_cache.get(cache_key)
        self.assertEqual(cached_response, response, "Cache should persist across instances")
    
    def test_cache_file_operations(self):
        """Cache file should be created and grow with entries."""
        # Initially no cache file or empty
        initial_exists = os.path.exists(self.cache.cache_file)
        
        # Add entry
        conversation = ["Test conversation"]
        cache_key = self.cache.get_cache_key(conversation)
        response = "Test response with substantial content " * 10
        self.cache.put(cache_key, response)
        
        # File should exist and have content
        self.assertTrue(os.path.exists(self.cache.cache_file), "Cache file should exist")
        self.assertGreater(os.path.getsize(self.cache.cache_file), 0, "Cache file should have content")
        
        # Should be retrievable
        cached_response = self.cache.get(cache_key)
        self.assertEqual(cached_response, response, "Should retrieve stored response")
    
    def test_empty_response_handling(self):
        """Empty responses should not be cached."""
        conversation = ["Test prompt"]
        cache_key = self.cache.get_cache_key(conversation)
        
        # Try to cache empty response
        self.cache.put(cache_key, "")
        
        # Should not be cached
        result = self.cache.get(cache_key)
        self.assertIsNone(result, "Empty responses should not be cached")
        
        # Try whitespace-only response
        self.cache.put(cache_key, "   \n  ")
        result = self.cache.get(cache_key)
        self.assertIsNone(result, "Whitespace-only responses should not be cached")
    
    def test_multiprocess_safety_simulation(self):
        """Simulate multiple processes accessing the same cache."""
        conversation = ["Multiprocess test"]
        cache_key = self.cache.get_cache_key(conversation)
        response = "Multiprocess response"
        
        # Store in first cache
        self.cache.put(cache_key, response)
        
        # Create second cache instance (simulates another worker)
        cache2 = LLMCache("test-model", cache_dir=self.temp_dir)
        
        # Should load existing cache
        cached_response = cache2.get(cache_key)
        self.assertEqual(cached_response, response, "Second instance should load existing cache")
        
        # Add entry in second cache
        conversation2 = ["Second process test"]
        cache_key2 = cache2.get_cache_key(conversation2)
        response2 = "Second process response"
        cache2.put(cache_key2, response2)
        
        # First cache should pick up changes after reload check
        # Force reload by creating new instance
        cache3 = LLMCache("test-model", cache_dir=self.temp_dir)
        cached_response2 = cache3.get(cache_key2)
        self.assertEqual(cached_response2, response2, "Changes should be visible across instances")
    
    def test_different_parameters_different_keys(self):
        """Different parameters should generate different cache keys."""
        conversation = ["Same conversation"]
        
        key1 = self.cache.get_cache_key(conversation, max_tokens=100)
        key2 = self.cache.get_cache_key(conversation, max_tokens=200)
        key3 = self.cache.get_cache_key(conversation, json=True)
        key4 = self.cache.get_cache_key(conversation)
        
        # All should be different
        keys = [key1, key2, key3, key4]
        unique_keys = set(keys)
        self.assertEqual(len(unique_keys), len(keys), "Different parameters should generate different keys")
    
    def test_json_mode_parameter(self):
        """JSON mode should be included in cache key."""
        conversation = ["Test prompt"]
        
        key_without_json = self.cache.get_cache_key(conversation)
        key_with_json = self.cache.get_cache_key(conversation, json=True)
        
        self.assertNotEqual(key_without_json, key_with_json, "JSON mode should create different key")
        self.assertIn(('json_mode', True), key_with_json, "JSON mode should be in key")
    
    def test_cache_miss_backward_compatibility(self):
        """Test backward compatibility scenarios that caused cache misses."""
        conversation = ["Test conversation for compatibility"]
        
        # Scenario 1: Legacy cache pattern (hparams only)
        legacy_key = self.cache.get_cache_key(
            conversation, 
            hparams={'max_tokens': 4096}
        )
        
        # Scenario 2: Current benchmark pattern (explicit max_tokens + hparams)
        current_key = self.cache.get_cache_key(
            conversation,
            max_tokens=4096,
            hparams={'max_tokens': 4096}
        )
        
        # Scenario 3: Vision test pattern (explicit max_tokens)
        vision_key = self.cache.get_cache_key(
            conversation,
            max_tokens=4096
        )
        
        # All should generate identical keys for same max_tokens value
        self.assertEqual(legacy_key, current_key, "Legacy and current patterns should match")
        self.assertEqual(legacy_key, vision_key, "Legacy and vision patterns should match")
        
        # Verify normalized structure
        expected_components = 2  # conversation + hparams
        self.assertEqual(len(legacy_key), expected_components, "Should have 2 components")
        self.assertIn(('hparams', (('max_tokens', 4096),)), legacy_key)
    
    def test_parameter_precedence(self):
        """Test that explicit parameters take precedence over hparams."""
        conversation = ["Test precedence"]
        
        # Explicit max_tokens should override hparams max_tokens
        key = self.cache.get_cache_key(
            conversation,
            max_tokens=8192,
            hparams={'max_tokens': 4096, 'temperature': 0.7}
        )
        
        # Should contain explicit value, not hparams value
        self.assertIn(('hparams', (('max_tokens', 8192), ('temperature', 0.7))), key)
        self.assertNotIn(('max_tokens', 4096), key)


if __name__ == '__main__':
    unittest.main(verbosity=2)