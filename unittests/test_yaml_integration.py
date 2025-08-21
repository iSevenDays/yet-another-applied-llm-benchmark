#!/usr/bin/env python3
"""
Integration test suite for YAML configuration files - TDD implementation.
Focus: Test actual config.yaml and config.yaml.example files work correctly.
"""
import unittest
import os
import yaml
from pathlib import Path
from config_loader import load_config


class TestYAMLConfigFiles(unittest.TestCase):
    """Test actual YAML configuration files in the project."""
    
    def setUp(self):
        """Set up test environment."""
        self.project_root = Path(__file__).parent.parent
        self.config_path = self.project_root / "config.yaml"
        self.example_config_path = self.project_root / "config.yaml.example"
    
    def test_config_yaml_exists_and_loads(self):
        """Test that config.yaml exists and loads correctly."""
        self.assertTrue(self.config_path.exists(), "config.yaml should exist")
        
        # Test loading with our config_loader
        config = load_config(str(self.config_path))
        
        # Validate required structure
        self.assertIn('container', config)
        self.assertIn('hparams', config)  
        self.assertIn('llms', config)
        self.assertIn('system_prompt', config)
        
        # Validate container is valid value
        self.assertIn(config['container'], ['docker', 'podman'])
        
        # Validate hparams is dict
        self.assertIsInstance(config['hparams'], dict)
        
        # Validate llms is dict
        self.assertIsInstance(config['llms'], dict)
    
    def test_config_yaml_thinking_budget_structure(self):
        """Test that thinking budget configuration is correctly structured."""
        config = load_config(str(self.config_path))
        
        # Check if OpenAI config has thinking budget
        if 'openai' in config['llms']:
            openai_config = config['llms']['openai']
            if 'hparams' in openai_config and 'extra_body' in openai_config['hparams']:
                extra_body = openai_config['hparams']['extra_body']
                self.assertIn('chat_template_kwargs', extra_body)
                self.assertIn('thinking_budget', extra_body['chat_template_kwargs'])
                
                thinking_budget = extra_body['chat_template_kwargs']['thinking_budget']
                self.assertIsInstance(thinking_budget, int)
                self.assertGreater(thinking_budget, 0)
    
    def test_example_config_yaml_exists_and_loads(self):
        """Test that config.yaml.example exists and loads correctly."""
        self.assertTrue(self.example_config_path.exists(), "config.yaml.example should exist")
        
        # Test loading with our config_loader
        config = load_config(str(self.example_config_path))
        
        # Validate required structure
        self.assertIn('container', config)
        self.assertIn('hparams', config)
        self.assertIn('llms', config)
        self.assertIn('system_prompt', config)
        
        # Example should have more comprehensive structure
        self.assertIsInstance(config['llms'], dict)
        # Example should contain multiple providers as examples
        self.assertGreater(len(config['llms']), 3, "Example should show multiple providers")
    
    def test_yaml_files_have_valid_syntax(self):
        """Test that YAML files have valid syntax when parsed with PyYAML directly."""
        # Test config.yaml
        with open(self.config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        self.assertIsInstance(config_data, dict)
        
        # Test config.yaml.example  
        with open(self.example_config_path, 'r') as f:
            example_data = yaml.safe_load(f)
        self.assertIsInstance(example_data, dict)
    
    def test_config_compatibility_with_model_expectations(self):
        """Test that YAML config structure matches what model files expect."""
        config = load_config(str(self.config_path))
        
        # Test structure that model files expect
        self.assertIsInstance(config, dict)
        
        # Test that we can access nested provider configs like model files do
        for provider_name, provider_config in config['llms'].items():
            self.assertIsInstance(provider_config, dict)
            
            # Most providers should have at least api_key, api_base, or project_id (for VertexAI)
            has_api_key = 'api_key' in provider_config
            has_api_base = 'api_base' in provider_config
            has_project_id = 'project_id' in provider_config  # VertexAI specific
            
            self.assertTrue(has_api_key or has_api_base or has_project_id, 
                          f"Provider {provider_name} should have api_key, api_base, or project_id")
    
    def test_global_hparams_structure(self):
        """Test that global hparams are properly structured."""
        config = load_config(str(self.config_path))
        
        hparams = config['hparams']
        self.assertIsInstance(hparams, dict)
        
        # Check common parameters
        if 'max_tokens' in hparams:
            self.assertIsInstance(hparams['max_tokens'], int)
            self.assertGreater(hparams['max_tokens'], 0)
        
        if 'temperature' in hparams:
            self.assertIsInstance(hparams['temperature'], (int, float))
            self.assertGreaterEqual(hparams['temperature'], 0.0)
            self.assertLessEqual(hparams['temperature'], 2.0)
    
    def test_provider_specific_hparams_inheritance(self):
        """Test that provider-specific hparams can override globals."""
        config = load_config(str(self.config_path))
        
        # Look for providers with their own hparams
        for provider_name, provider_config in config['llms'].items():
            if 'hparams' in provider_config:
                provider_hparams = provider_config['hparams']
                self.assertIsInstance(provider_hparams, dict)
                
                # Provider hparams should be able to contain any valid parameters
                # This test just ensures the structure is correct
                for param_name, param_value in provider_hparams.items():
                    self.assertIsInstance(param_name, str)
                    # param_value can be any type depending on parameter


class TestYAMLConfigMigrationCompatibility(unittest.TestCase):
    """Test that YAML config maintains compatibility with existing JSON usage patterns."""
    
    def setUp(self):
        """Set up test environment."""
        self.project_root = Path(__file__).parent.parent
        self.config_path = self.project_root / "config.yaml"
    
    def test_json_style_access_patterns(self):
        """Test that config can be accessed like the old JSON config."""
        config = load_config(str(self.config_path))
        
        # Test access patterns that model files use
        container_backend = config['container']
        self.assertIsInstance(container_backend, str)
        
        global_hparams = config['hparams'] 
        self.assertIsInstance(global_hparams, dict)
        
        llm_configs = config['llms']
        self.assertIsInstance(llm_configs, dict)
        
        # Test nested access patterns
        for provider_name in llm_configs:
            provider_config = llm_configs[provider_name]
            self.assertIsInstance(provider_config, dict)
            
            # Test .get() patterns that model files use
            api_key = provider_config.get('api_key')
            if api_key is not None:
                self.assertIsInstance(api_key, str)
            
            provider_hparams = provider_config.get('hparams', {})
            self.assertIsInstance(provider_hparams, dict)
    
    def test_backward_compatibility_structure(self):
        """Test that YAML config has same structure as old JSON config."""
        config = load_config(str(self.config_path))
        
        # Required top-level keys that existing code expects
        required_keys = ['container', 'hparams', 'llms']
        for key in required_keys:
            self.assertIn(key, config, f"Missing required key: {key}")
        
        # Optional keys that code might expect
        optional_keys = ['system_prompt']
        for key in optional_keys:
            self.assertIn(key, config, f"Missing optional key: {key}")
        
        # Validate types match expectations
        self.assertIsInstance(config['container'], str)
        self.assertIsInstance(config['hparams'], dict) 
        self.assertIsInstance(config['llms'], dict)
        self.assertIsInstance(config['system_prompt'], str)


if __name__ == '__main__':
    unittest.main()