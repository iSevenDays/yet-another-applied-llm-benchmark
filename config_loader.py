#!/usr/bin/env python3
"""
YAML Configuration Loader - Simple, focused config loading for LLM benchmark.
SPARC-compliant: Simple interface, no fallback complexity, clean error handling.
"""
import yaml
import os
from typing import Dict, Any, Optional


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load YAML configuration file with validation and sensible defaults.
    
    Args:
        config_path: Path to YAML config file. If None, looks for 'config.yaml' 
                    in current directory.
    
    Returns:
        Dictionary containing parsed configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML is malformed
        ValueError: If config is invalid (empty, missing required fields)
    """
    # Determine config file path
    if config_path is None:
        config_path = 'config.yaml'
    
    # Check if file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load and parse YAML
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in {config_path}: {e}")
    
    # Handle empty file
    if config is None:
        raise ValueError(f"Configuration file is empty: {config_path}")
    
    # Validate required fields
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a dictionary, got {type(config)}")
    
    if 'container' not in config:
        raise ValueError("Missing required field 'container' in configuration")
    
    # Provide sensible defaults for optional fields
    config.setdefault('hparams', {})
    config.setdefault('llms', {})
    config.setdefault('system_prompt', '')
    
    return config


def get_config() -> Dict[str, Any]:
    """
    Convenience function to load config.yaml from current directory.
    Equivalent to load_config() with no arguments.
    """
    return load_config()


# For backward compatibility - some files might expect this pattern
def load_json_config():
    """
    DEPRECATED: For migration period only.
    Loads YAML config but provides same interface as old JSON loader.
    """
    import warnings
    warnings.warn(
        "load_json_config() is deprecated. Use load_config() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return load_config()