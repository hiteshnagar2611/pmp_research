"""
Configuration utilities: loading, saving, and merging configs.

Supports:
  - YAML configuration files
  - Dict-based configs
  - Config merging and updates
  - Type conversion
"""

from __future__ import annotations

from typing import Dict, Any, Optional
from pathlib import Path
import json
import yaml


def load_config(config_file: str) -> Dict[str, Any]:
    """
    Load configuration from file.

    Supports YAML and JSON formats.

    Args:
        config_file: path to config file

    Returns:
        config: configuration dictionary
    """
    config_path = Path(config_file)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    if config_path.suffix in ['.yaml', '.yml']:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    elif config_path.suffix == '.json':
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")

    return config or {}


def save_config(config: Dict[str, Any], output_file: str):
    """
    Save configuration to file.

    Args:
        config: configuration dictionary
        output_file: output file path
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix in ['.yaml', '.yml']:
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    elif output_path.suffix == '.json':
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
    else:
        raise ValueError(f"Unsupported config format: {output_path.suffix}")


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """
    Merge two configuration dictionaries.

    Recursively merges, with override_config taking precedence.

    Args:
        base_config: base configuration
        override_config: configuration to override with

    Returns:
        merged: merged configuration
    """
    merged = base_config.copy()

    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged


def flatten_config(config: Dict, prefix: str = '') -> Dict[str, Any]:
    """
    Flatten nested configuration dictionary.

    Args:
        config: nested configuration
        prefix: key prefix for flattened keys

    Returns:
        flattened: flat configuration with dot-notation keys
    """
    flattened = {}

    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            flattened.update(flatten_config(value, full_key))
        else:
            flattened[full_key] = value

    return flattened


def unflatten_config(flat_config: Dict[str, Any]) -> Dict:
    """
    Unflatten configuration dictionary.

    Args:
        flat_config: flat configuration with dot-notation keys

    Returns:
        config: nested configuration
    """
    config = {}

    for key, value in flat_config.items():
        parts = key.split('.')
        current = config

        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[parts[-1]] = value

    return config


def update_config_from_args(config: Dict, args: Dict) -> Dict:
    """
    Update config from command-line arguments.

    Args:
        config: configuration dictionary
        args: argument dictionary (can use dot-notation)

    Returns:
        updated: updated configuration
    """
    updated = config.copy()

    flat_config = flatten_config(updated)
    flat_config.update(args)
    updated = unflatten_config(flat_config)

    return updated


class Config:
    """Configuration object with attribute access."""

    def __init__(self, config_dict: Optional[Dict] = None):
        """
        Initialize config object.

        Args:
            config_dict: configuration dictionary
        """
        if config_dict is None:
            config_dict = {}

        self._config = config_dict

    def __getattr__(self, key: str) -> Any:
        """Get attribute from config."""
        if key.startswith('_'):
            return super().__getattribute__(key)

        if key in self._config:
            value = self._config[key]
            if isinstance(value, dict):
                return Config(value)
            return value

        raise AttributeError(f"Config has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any):
        """Set attribute in config."""
        if key.startswith('_'):
            super().__setattr__(key, value)
        else:
            self._config[key] = value

    def __getitem__(self, key: str) -> Any:
        """Get item from config."""
        value = self._config[key]
        if isinstance(value, dict):
            return Config(value)
        return value

    def __setitem__(self, key: str, value: Any):
        """Set item in config."""
        self._config[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get item with default."""
        if key in self._config:
            return self._config[key]
        return default

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return self._config.copy()

    def update(self, other: Dict):
        """Update config from dictionary."""
        self._config.update(other)

    def __str__(self) -> str:
        """String representation."""
        return json.dumps(self._config, indent=2)

    def __repr__(self) -> str:
        """Repr."""
        return f"Config({self._config})"


if __name__ == "__main__":
    # Test config management
    print("Config utilities loaded successfully")

    # Test flatten/unflatten
    config = {
        'model': {
            'type': 'dynamo',
            'hidden_dim': 256,
        },
        'train': {
            'lr': 1e-4,
        }
    }

    flattened = flatten_config(config)
    print(f"Flattened: {flattened}")

    unflattened = unflatten_config(flattened)
    print(f"Unflattened: {unflattened}")

    # Test Config object
    cfg = Config(config)
    print(f"Config.model.type: {cfg.model.type}")
    print(f"Config.train.lr: {cfg.train.lr}")

    print("✓ Config utilities working!")
