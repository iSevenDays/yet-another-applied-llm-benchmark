#!/usr/bin/env python3
"""
Test suite for YAML configuration loader - TDD implementation.
Focus: Comprehensive coverage of config loading, validation, and error handling.
"""
import unittest
import tempfile
import os
import yaml
from pathlib import Path


class TestYAMLConfigLoader(unittest.TestCase):
    """Test the YAML configuration loader with TDD approach."""
    
    def setUp(self):
        """Set up test environment with temporary directories."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "config.yaml")
        
    def tearDown(self):
        """Clean up test files."""
        if os.path.exists(self.config_path):
            os.remove(self.config_path)
        os.rmdir(self.temp_dir)
    
    def create_test_config(self, config_data):
        """Helper to create test YAML config file."""
        with open(self.config_path, 'w') as f:
            yaml.dump(config_data, f)
    
    def test_load_valid_config(self):
        """Test loading a valid YAML configuration."""
        config_data = {
            'container': 'docker',
            'hparams': {'max_tokens': 4096},
            'llms': {
                'openai': {
                    'api_key': 'test-key',
                    'api_base': 'https://api.openai.com/v1'
                }
            }
        }
        self.create_test_config(config_data)
        
        # This will fail until we implement load_config()
        from config_loader import load_config
        result = load_config(self.config_path)
        
        self.assertEqual(result['container'], 'docker')
        self.assertEqual(result['hparams']['max_tokens'], 4096)
        self.assertEqual(result['llms']['openai']['api_key'], 'test-key')
    
    def test_load_config_with_thinking_budget(self):
        """Test loading config with thinking budget parameters."""
        config_data = {
            'container': 'docker',
            'hparams': {'max_tokens': 4096},
            'llms': {
                'openai': {
                    'api_key': 'test-key',
                    'api_base': 'http://localhost:8000/v1',
                    'hparams': {
                        'extra_body': {
                            'chat_template_kwargs': {
                                'thinking_budget': 20000
                            }
                        }
                    }
                }
            }
        }
        self.create_test_config(config_data)
        
        from config_loader import load_config
        result = load_config(self.config_path)
        
        thinking_budget = result['llms']['openai']['hparams']['extra_body']['chat_template_kwargs']['thinking_budget']
        self.assertEqual(thinking_budget, 20000)
    
    def test_load_minimal_config(self):
        """Test loading minimal configuration with required fields only."""
        config_data = {
            'container': 'podman',
            'llms': {}
        }
        self.create_test_config(config_data)
        
        from config_loader import load_config
        result = load_config(self.config_path)
        
        self.assertEqual(result['container'], 'podman')
        self.assertEqual(result['llms'], {})
        # Should handle missing hparams gracefully
        self.assertIn('hparams', result)  # Should provide default empty dict
    
    def test_load_config_with_defaults(self):
        """Test that missing optional fields get sensible defaults."""
        config_data = {
            'container': 'docker',
            'llms': {
                'anthropic': {'api_key': 'test-key'}
            }
        }
        self.create_test_config(config_data)
        
        from config_loader import load_config
        result = load_config(self.config_path)
        
        # Should provide default empty hparams
        self.assertIsInstance(result.get('hparams', {}), dict)
        # Should handle missing system_prompt
        self.assertIn('system_prompt', result)
    
    def test_load_config_missing_file(self):
        """Test error handling when config file doesn't exist."""
        from config_loader import load_config
        
        with self.assertRaises(FileNotFoundError) as context:
            load_config('/nonexistent/config.yaml')
        
        self.assertIn('config.yaml', str(context.exception))
    
    def test_load_config_invalid_yaml(self):
        """Test error handling for malformed YAML."""
        with open(self.config_path, 'w') as f:
            f.write("invalid: yaml: content: [\nbroken")
        
        from config_loader import load_config
        
        with self.assertRaises(yaml.YAMLError):
            load_config(self.config_path)
    
    def test_load_config_empty_file(self):
        """Test handling of empty YAML file."""
        with open(self.config_path, 'w') as f:
            f.write("")
        
        from config_loader import load_config
        
        with self.assertRaises(ValueError) as context:
            load_config(self.config_path)
        
        self.assertIn('empty', str(context.exception).lower())
    
    def test_load_config_missing_required_fields(self):
        """Test validation of required configuration fields."""
        # Missing 'container' field
        config_data = {
            'llms': {'openai': {'api_key': 'test'}}
        }
        self.create_test_config(config_data)
        
        from config_loader import load_config
        
        with self.assertRaises(ValueError) as context:
            load_config(self.config_path)
        
        self.assertIn('container', str(context.exception))
    
    def test_load_config_with_comments(self):
        """Test that YAML comments are preserved in parsing."""
        yaml_content = """
# Container backend configuration
container: docker

# Global hyperparameters
hparams:
  max_tokens: 4096  # Maximum tokens per response
  temperature: 0.7  # Sampling temperature

# Model provider settings
llms:
  openai:
    api_key: "test-key"
    # Custom API endpoint
    api_base: "http://localhost:8000/v1"
"""
        with open(self.config_path, 'w') as f:
            f.write(yaml_content)
        
        from config_loader import load_config
        result = load_config(self.config_path)
        
        # Comments should not affect parsing
        self.assertEqual(result['container'], 'docker')
        self.assertEqual(result['hparams']['temperature'], 0.7)
    
    def test_load_config_complex_structure(self):
        """Test loading complex nested configuration structure."""
        config_data = {
            'container': 'docker',
            'system_prompt': 'You are a helpful assistant',
            'hparams': {
                'max_tokens': 4096,
                'temperature': 0.7,
                'top_p': 0.9
            },
            'llms': {
                'openai': {
                    'api_key': 'key1',
                    'api_base': 'http://localhost:8000/v1',
                    'hparams': {
                        'temperature': 1.0,  # Override global
                        'extra_body': {
                            'chat_template_kwargs': {
                                'thinking_budget': 15000
                            }
                        }
                    }
                },
                'anthropic': {
                    'api_key': 'key2',
                    'hparams': {
                        'max_tokens': 8192  # Override global
                    }
                },
                'ollama': {
                    'api_base': 'http://192.168.1.100:11434'
                }
            }
        }
        self.create_test_config(config_data)
        
        from config_loader import load_config
        result = load_config(self.config_path)
        
        # Verify complex nested structure
        self.assertEqual(result['system_prompt'], 'You are a helpful assistant')
        self.assertEqual(result['llms']['openai']['hparams']['temperature'], 1.0)
        self.assertEqual(result['llms']['anthropic']['hparams']['max_tokens'], 8192)
        self.assertEqual(result['llms']['ollama']['api_base'], 'http://192.168.1.100:11434')
    
    def test_default_config_path_detection(self):
        """Test that load_config() can find config.yaml in current directory."""
        # Create config.yaml in temp dir and change to that directory
        original_cwd = os.getcwd()
        try:
            os.chdir(self.temp_dir)
            config_data = {'container': 'docker', 'llms': {}}
            with open('config.yaml', 'w') as f:
                yaml.dump(config_data, f)
            
            from config_loader import load_config
            # Should find config.yaml in current directory when no path specified
            result = load_config()
            self.assertEqual(result['container'], 'docker')
        finally:
            os.chdir(original_cwd)


class TestConfigLoaderIntegration(unittest.TestCase):
    """Integration tests for config loader with model system."""
    
    def test_config_structure_matches_existing_json(self):
        """Test that YAML config structure matches existing JSON expectations."""
        # This ensures the YAML format produces same structure as current JSON
        yaml_config = {
            'container': 'docker',
            'system_prompt': '',
            'hparams': {'max_tokens': 4096},
            'llms': {
                'openai': {
                    'api_key': 'sk-test',
                    'api_base': 'https://api.openai.com/v1',
                    'hparams': {
                        'extra_body': {
                            'chat_template_kwargs': {
                                'thinking_budget': 20000
                            }
                        }
                    }
                }
            }
        }
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        try:
            yaml.dump(yaml_config, temp_file)
            temp_file.close()
            
            from config_loader import load_config
            result = load_config(temp_file.name)
            
            # Verify structure matches what model files expect
            self.assertIn('container', result)
            self.assertIn('hparams', result)
            self.assertIn('llms', result)
            self.assertIsInstance(result['llms'], dict)
            self.assertIn('openai', result['llms'])
            
        finally:
            os.unlink(temp_file.name)


if __name__ == '__main__':
    unittest.main()