"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Configuration loading and management system.

This module provides configuration loading, validation, and default
value management for the 7D phase field theory project.

Physical Meaning:
    Configuration management ensures proper parameter hierarchy and
    validation for phase field simulations, maintaining physical
    consistency across different experimental setups.

Mathematical Foundation:
    Configuration parameters control the fractional Riesz operator
    L_β = μ(-Δ)^β + λ and related physical constants.

Example:
    >>> loader = ConfigLoader()
    >>> config = loader.load_config("configs/default.json")
    >>> solver_params = config['solver']
"""

import json
import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path
import os


class ConfigLoader:
    """
    Configuration loader for the 7D phase field theory project.

    Physical Meaning:
        Manages configuration parameters for the 7D phase field theory
        simulations, ensuring proper parameter hierarchy and validation.

    Priority Order:
        1. CLI arguments (highest priority)
        2. Environment variables
        3. Configuration file (lowest priority)
    """

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize configuration loader.

        Physical Meaning:
            Sets up the configuration loader with default directory
            and loads default configuration values.

        Args:
            config_dir (Optional[Path]): Directory containing configuration files.
        """
        self.config_dir = config_dir or Path("configs")
        self._defaults = self._load_defaults()

    def load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load configuration from file.

        Physical Meaning:
            Loads configuration parameters from file, merging with
            defaults and validating physical consistency.

        Args:
            config_path (Union[str, Path]): Path to configuration file.

        Returns:
            Dict[str, Any]: Configuration dictionary with validated parameters.

        Raises:
            ValueError: If configuration format is unsupported.
            FileNotFoundError: If configuration file doesn't exist.
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Load configuration based on file extension
        if config_path.suffix == ".json":
            with open(config_path, "r") as f:
                config = json.load(f)
        elif config_path.suffix in [".yaml", ".yml"]:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")

        # Merge with defaults
        config = self._merge_with_defaults(config)

        # Validate configuration
        self._validate_config(config)

        return config

    def get_parameter(self, key: str, default: Any = None) -> Any:
        """
        Get parameter value following priority order.

        Physical Meaning:
            Retrieves configuration parameters in the correct priority order,
            ensuring that CLI arguments override environment variables,
            which override configuration file values.

        Args:
            key (str): Parameter name.
            default (Any): Default value if parameter not found.

        Returns:
            Any: Parameter value following priority order.
        """
        # 1. Check CLI arguments (highest priority)
        cli_value = self._get_cli_argument(key)
        if cli_value is not None:
            return cli_value

        # 2. Check environment variables
        env_value = self._get_environment_variable(key)
        if env_value is not None:
            return env_value

        # 3. Check configuration file
        config_value = self._defaults.get(key)
        if config_value is not None:
            return config_value

        # 4. Return default value
        return default

    def _load_defaults(self) -> Dict[str, Any]:
        """
        Load default configuration values.

        Physical Meaning:
            Provides default values for all configuration parameters,
            ensuring physical consistency and reasonable defaults.

        Returns:
            Dict[str, Any]: Default configuration dictionary.
        """
        return {
            "domain": {
                "L": 1.0,
                "N": 256,
                "dimensions": 3,
                "N_phi": 32,
                "N_t": 64,
                "T": 1.0,
            },
            "physics": {
                "mu": 1.0,
                "beta": 1.0,
                "lambda": 0.0,
                "nu": 1.0,
                "phase_velocity": 1e15,
            },
            "solver": {
                "precision": "float64",
                "fft_plan": "MEASURE",
                "tolerance": 1e-12,
                "max_iterations": 1000,
            },
            "output": {
                "save_fields": True,
                "save_spectra": True,
                "save_analysis": True,
                "format": "hdf5",
                "output_dir": "output",
            },
            "visualization": {
                "plot_format": "png",
                "dpi": 300,
                "style": "seaborn-v0_8",
                "color_map": "viridis",
            },
            "analysis": {
                "compute_energy": True,
                "compute_topology": True,
                "compute_coherence": True,
                "save_metrics": True,
            },
        }

    def _merge_with_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge configuration with defaults.

        Physical Meaning:
            Recursively merges configuration with default values,
            ensuring all required parameters are present.

        Args:
            config (Dict[str, Any]): Configuration to merge.

        Returns:
            Dict[str, Any]: Merged configuration.
        """

        def deep_merge(
            default: Dict[str, Any], override: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Recursively merge dictionaries."""
            result = default.copy()
            for key, value in override.items():
                if (
                    key in result
                    and isinstance(result[key], dict)
                    and isinstance(value, dict)
                ):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result

        return deep_merge(self._defaults, config)

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate configuration parameters.

        Physical Meaning:
            Validates that configuration parameters have physically
            reasonable values and are consistent with each other.

        Args:
            config (Dict[str, Any]): Configuration to validate.

        Raises:
            ValueError: If configuration is invalid.
        """
        # Validate domain parameters
        domain = config.get("domain", {})
        if domain.get("L", 0) <= 0:
            raise ValueError("Domain size L must be positive")
        if domain.get("N", 0) <= 0:
            raise ValueError("Number of grid points N must be positive")
        if domain.get("dimensions", 0) not in [1, 2, 3]:
            raise ValueError("Dimensions must be 1, 2, or 3")

        # Validate physics parameters
        physics = config.get("physics", {})
        if physics.get("mu", 0) <= 0:
            raise ValueError("Diffusion coefficient mu must be positive")
        if not (0 < physics.get("beta", 0) < 2):
            raise ValueError("Fractional order beta must be in (0, 2)")
        if physics.get("lambda", 0) < 0:
            raise ValueError("Damping parameter lambda must be non-negative")

        # Validate solver parameters
        solver = config.get("solver", {})
        if solver.get("tolerance", 0) <= 0:
            raise ValueError("Solver tolerance must be positive")
        if solver.get("max_iterations", 0) <= 0:
            raise ValueError("Maximum iterations must be positive")

    def _get_cli_argument(self, key: str) -> Optional[Any]:
        """
        Get CLI argument value.

        Physical Meaning:
            Retrieves parameter value from command line arguments,
            providing highest priority override.

        Args:
            key (str): Parameter name.

        Returns:
            Optional[Any]: CLI argument value if found.
        """
        # This would be implemented to check actual CLI arguments
        # For now, return None
        return None

    def _get_environment_variable(self, key: str) -> Optional[Any]:
        """
        Get environment variable value.

        Physical Meaning:
            Retrieves parameter value from environment variables,
            providing medium priority override.

        Args:
            key (str): Parameter name.

        Returns:
            Optional[Any]: Environment variable value if found.
        """
        # Convert key to environment variable format
        env_key = f"BHLFF_{key.upper()}"
        value = os.getenv(env_key)

        if value is not None:
            # Try to convert to appropriate type
            try:
                # Try integer
                if value.isdigit():
                    return int(value)
                # Try float
                try:
                    return float(value)
                except ValueError:
                    pass
                # Try boolean
                if value.lower() in ["true", "false"]:
                    return value.lower() == "true"
                # Return as string
                return value
            except ValueError:
                return value

        return None

    def save_config(
        self, config: Dict[str, Any], config_path: Union[str, Path]
    ) -> None:
        """
        Save configuration to file.

        Physical Meaning:
            Saves configuration parameters to file for reuse
            and reproducibility of experiments.

        Args:
            config (Dict[str, Any]): Configuration to save.
            config_path (Union[str, Path]): Path to save configuration.
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        if config_path.suffix == ".json":
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
        elif config_path.suffix in [".yaml", ".yml"]:
            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")

    def __repr__(self) -> str:
        """String representation of the configuration loader."""
        return f"ConfigLoader(config_dir={self.config_dir})"
