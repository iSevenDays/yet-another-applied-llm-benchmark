#!/usr/bin/env python3
"""
Test suite for YAML config integration with caching system - TDD approach.
Focus: Reproduce and fix "unhashable type: 'dict'" error with thinking budget.
"""
import unittest
import tempfile
import os
from llm_cache import LLMCache
from config_loader import load_config


class TestYAMLCacheIntegration(unittest.TestCase):
    """Test YAML config integration with caching system."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = LLMCache("test-model", cache_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_cache_key_with_nested_dict_hparams_works(self):
        """Test that fixed cache system works with nested dict hparams (TDD Green)."""
        # This reproduces the "unhashable type: 'dict'" error
        conversation = ["Test prompt"]
        
        # This is the problematic hparams structure from YAML config
        hparams_with_nested_dict = {
            'max_tokens': 4096,
            'extra_body': {
                'chat_template_kwargs': {
                    'thinking_budget': 32768
                }
            }
        }
        
        # Cache key generation should succeed
        cache_key = self.cache.get_cache_key(
            conversation, 
            hparams=hparams_with_nested_dict
        )
        
        # Using it as dict key should now work (after our fix)
        test_dict = {}
        test_dict[cache_key] = "test_value"  # This should work now
        
        # Verify the cache key is hashable
        hash_value = hash(cache_key)
        self.assertIsInstance(hash_value, int)
        
        # Verify we can retrieve the value
        self.assertEqual(test_dict[cache_key], "test_value")
    
    def test_cache_key_with_simple_hparams_works(self):
        """Test that cache system works with simple (non-nested) hparams."""
        conversation = ["Test prompt"]
        
        # Simple hparams should work fine
        simple_hparams = {
            'max_tokens': 4096,
            'temperature': 0.7
        }
        
        # This should work without errors
        cache_key = self.cache.get_cache_key(
            conversation, 
            hparams=simple_hparams
        )
        
        self.assertIsInstance(cache_key, tuple)
    
    def test_real_yaml_config_works_with_cache(self):
        """Test that real YAML config with thinking budget works with caching."""
        # Load the actual YAML config
        config = load_config()
        
        # Extract thinking budget hparams like the model files do
        if 'openai' in config['llms'] and 'hparams' in config['llms']['openai']:
            openai_hparams = config['llms']['openai']['hparams']
            
            # Merge global hparams like the model initialization does
            merged_hparams = config['hparams'].copy()
            merged_hparams.update(openai_hparams)
            
            conversation = ["Test prompt for real config"]
            
            # This should now work with our fix
            cache_key = self.cache.get_cache_key(
                conversation,
                hparams=merged_hparams
            )
            
            # Verify the cache key is usable
            test_dict = {}
            test_dict[cache_key] = "test_response"
            self.assertEqual(test_dict[cache_key], "test_response")
    
    def test_llm_initialization_with_yaml_config_integration(self):
        """Test that LLM initialization with YAML config integrates with caching."""
        try:
            from llm import LLM
            
            # Create LLM with YAML config that has thinking budget
            llm = LLM('openai_gpt-3.5-turbo')
            
            # Try to generate a cache key - this should now work with our fix
            if hasattr(llm, 'cache') and llm.cache:
                conversation = ["Test message"]
                
                # This attempts to use the actual hparams from YAML config
                cache_key = llm.cache.get_cache_key(
                    conversation,
                    hparams=llm.model.hparams
                )
                
                # Verify the cache key works
                test_dict = {}
                test_dict[cache_key] = "test_response"
                self.assertEqual(test_dict[cache_key], "test_response")
                
        except ImportError:
            self.skipTest("LLM class not available for integration test")


class TestCacheKeyGeneration(unittest.TestCase):
    """Test cache key generation with different data structures."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = LLMCache("test-model", cache_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_cache_key_generation_edge_cases(self):
        """Test cache key generation with various complex structures now works."""
        conversation = ["Test"]
        
        # Test cases that should now work with our fix
        complex_hparams = [
            # Nested dictionaries
            {'extra_body': {'key': 'value'}},
            
            # Lists (also unhashable when nested)
            {'stop_sequences': ['<|end|>', '<|stop|>']},
            
            # Mixed complex structures
            {
                'extra_body': {
                    'chat_template_kwargs': {
                        'thinking_budget': 32768,
                        'stop_sequences': ['<|thinking|>', '</thinking>']
                    }
                },
                'temperature': 0.7
            }
        ]
        
        for i, hparams in enumerate(complex_hparams):
            with self.subTest(case=i):
                # This should now work
                cache_key = self.cache.get_cache_key(conversation, hparams=hparams)
                
                # Verify the cache key is hashable and usable
                test_dict = {}
                test_dict[cache_key] = f"test_value_{i}"
                self.assertEqual(test_dict[cache_key], f"test_value_{i}")
                
                # Verify it's actually hashable
                hash_value = hash(cache_key)
                self.assertIsInstance(hash_value, int)


if __name__ == '__main__':
    unittest.main()