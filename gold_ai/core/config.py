"""Core configuration management for GOLD AI."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str = "config/settings.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Override with environment variables where applicable
    config = _apply_env_overrides(config)
    
    return config


def _apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply environment variable overrides to configuration.
    
    Environment variables follow pattern: GOLD_AI_<SECTION>_<KEY>
    Example: GOLD_AI_LOGGING_LEVEL=DEBUG
    """
    env_prefix = "GOLD_AI_"
    
    for key, value in os.environ.items():
        if key.startswith(env_prefix):
            # Parse nested key: GOLD_AI_DATA_BASE_DIR -> data.base_dir
            parts = key[len(env_prefix):].lower().split('_')
            
            # Navigate config dict and set value
            current = config
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # Try to parse as int/float/bool, otherwise keep as string
            final_key = parts[-1]
            current[final_key] = _parse_env_value(value)
    
    return config


def _parse_env_value(value: str) -> Any:
    """Parse environment variable value to appropriate type."""
    # Boolean
    if value.lower() in ('true', 'false'):
        return value.lower() == 'true'
    
    # Integer
    try:
        return int(value)
    except ValueError:
        pass
    
    # Float
    try:
        return float(value)
    except ValueError:
        pass
    
    # String (default)
    return value


def get_api_key(key_name: str, default: str = "") -> str:
    """Get API key from environment or secure config.
    
    NEVER hard-code API keys in source code!
    
    Args:
        key_name: Name of the API key (e.g., "NEWSAPI_KEY")
        default: Default value if not found
        
    Returns:
        API key string or empty string
    """
    return os.environ.get(key_name, default)


def validate_config(config: Dict[str, Any]) -> list[str]:
    """Validate configuration structure and values.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Required sections
    required_sections = ['app', 'data', 'logging', 'providers', 'agents']
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required config section: {section}")
    
    # Validate app mode
    if 'app' in config and 'mode' in config['app']:
        valid_modes = ['live', 'backtest', 'paper']
        if config['app']['mode'] not in valid_modes:
            errors.append(f"Invalid app mode: {config['app']['mode']}")
    
    # Validate logging level
    if 'logging' in config and 'level' in config['logging']:
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if config['logging']['level'].upper() not in valid_levels:
            errors.append(f"Invalid logging level: {config['logging']['level']}")
    
    # Validate provider settings
    if 'providers' in config:
        providers = config['providers']
        
        # Check that enabled providers have required config
        for name, provider in providers.items():
            if isinstance(provider, dict) and provider.get('enabled', False):
                if 'provider' not in provider:
                    errors.append(f"Provider '{name}' is enabled but has no provider type")
    
    # Validate agent weights sum approximately to 1.0
    if 'weighting' in config and 'base_weights' in config['weighting']:
        total_weight = sum(config['weighting']['base_weights'].values())
        if abs(total_weight - 1.0) > 0.1:
            errors.append(f"Base weights sum to {total_weight}, expected ~1.0")
    
    return errors
