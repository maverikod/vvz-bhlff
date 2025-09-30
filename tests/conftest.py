"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Pytest configuration and fixtures for the 7D phase field theory project.

This module provides pytest configuration and fixtures for testing
the BHLFF framework, including domains, fields, and parameters.

Physical Meaning:
    Test fixtures provide consistent test data for validating
    the 7D phase field theory implementation across different
    experimental setups and parameter ranges.

Example:
    >>> pytest tests/ -v
"""

import pytest
import numpy as np
from pathlib import Path
from typing import Dict, Any, Generator

from bhlff.core.domain.domain import Domain
from bhlff.core.domain.field import Field
from bhlff.core.phase.phase_field import PhaseField


@pytest.fixture
def test_domain() -> Domain:
    """
    Test domain configuration.
    
    Physical Meaning:
        Provides a standard 3D domain for testing phase field
        simulations with reasonable resolution and size.
        
    Returns:
        Domain: Test domain configuration.
    """
    return Domain(
        L=1.0,
        N=32,
        dimensions=3,
        N_phi=16,
        N_t=32,
        T=0.5
    )


@pytest.fixture
def test_domain_7d() -> Domain:
    """
    Test 7D domain configuration.
    
    Physical Meaning:
        Provides a 7D domain for testing full phase field
        simulations including phase and temporal dimensions.
        
    Returns:
        Domain: 7D test domain configuration.
    """
    return Domain(
        L=1.0,
        N=16,
        dimensions=3,
        N_phi=8,
        N_t=16,
        T=0.25
    )


@pytest.fixture
def test_physics_params() -> Dict[str, Any]:
    """
    Test physics parameters.
    
    Physical Meaning:
        Provides standard physics parameters for testing
        the fractional Riesz operator and related equations.
        
    Returns:
        Dict[str, Any]: Test physics parameters.
    """
    return {
        "mu": 1.0,
        "beta": 1.0,
        "lambda": 0.1,
        "nu": 1.0,
        "phase_velocity": 1e15
    }


@pytest.fixture
def test_solver_params() -> Dict[str, Any]:
    """
    Test solver parameters.
    
    Physical Meaning:
        Provides standard solver parameters for testing
        numerical methods and convergence.
        
    Returns:
        Dict[str, Any]: Test solver parameters.
    """
    return {
        "precision": "float64",
        "fft_plan": "MEASURE",
        "tolerance": 1e-10,
        "max_iterations": 100
    }


@pytest.fixture
def test_source_field(test_domain: Domain) -> np.ndarray:
    """
    Test source field.
    
    Physical Meaning:
        Creates a Gaussian source field for testing solver
        response and field evolution.
        
    Args:
        test_domain: Test domain configuration.
        
    Returns:
        np.ndarray: Test source field.
    """
    N = test_domain.N
    # Create a simple Gaussian source
    x = np.linspace(0, test_domain.L, N, endpoint=False)
    y = np.linspace(0, test_domain.L, N, endpoint=False)
    z = np.linspace(0, test_domain.L, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    center = test_domain.L / 2
    sigma = test_domain.L / 8
    
    source = np.exp(-((X - center)**2 + (Y - center)**2 + (Z - center)**2) / (2 * sigma**2))
    return source.astype(np.complex128)


@pytest.fixture
def test_phase_field(test_domain: Domain) -> PhaseField:
    """
    Test phase field.
    
    Physical Meaning:
        Creates a test phase field with initial conditions
        for testing phase field operations and evolution.
        
    Args:
        test_domain: Test domain configuration.
        
    Returns:
        PhaseField: Test phase field.
    """
    # Create initial field data
    field_data = np.zeros(test_domain.shape, dtype=np.complex128)
    
    # Add a simple initial condition
    center = test_domain.N // 2
    field_data[center, center, center] = 1.0 + 0.1j
    
    return PhaseField(
        data=field_data,
        domain=test_domain,
        time=0.0,
        phase_velocity=1e15
    )


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """
    Temporary output directory for tests.
    
    Physical Meaning:
        Provides a temporary directory for test output files,
        ensuring clean test environment and proper cleanup.
        
    Args:
        tmp_path: Pytest temporary path fixture.
        
    Yields:
        Path: Temporary output directory.
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    yield output_dir
    # Cleanup is handled by tmp_path fixture


@pytest.fixture(scope="session")
def reference_solutions() -> Dict[str, np.ndarray]:
    """
    Reference solutions for validation tests.
    
    Physical Meaning:
        Provides reference solutions for validating solver
        accuracy and convergence against known analytical
        or high-precision numerical solutions.
        
    Returns:
        Dict[str, np.ndarray]: Reference solutions.
    """
    # Load or generate reference solutions
    # This would typically load from a file or generate analytically
    return {}


@pytest.fixture
def test_config_loader():
    """
    Test configuration loader.
    
    Physical Meaning:
        Provides a configuration loader for testing
        configuration management and validation.
        
    Returns:
        ConfigLoader: Test configuration loader.
    """
    from bhlff.utils.config.loader import ConfigLoader
    return ConfigLoader()


@pytest.fixture
def test_logger():
    """
    Test structured logger.
    
    Physical Meaning:
        Provides a structured logger for testing
        logging functionality and output.
        
    Returns:
        StructuredLogger: Test logger.
    """
    from bhlff.utils.logging import StructuredLogger
    return StructuredLogger("test_logger")


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "level_a: marks tests for level A functionality"
    )
    config.addinivalue_line(
        "markers", "level_b: marks tests for level B functionality"
    )
    config.addinivalue_line(
        "markers", "level_c: marks tests for level C functionality"
    )
    config.addinivalue_line(
        "markers", "level_d: marks tests for level D functionality"
    )
    config.addinivalue_line(
        "markers", "level_e: marks tests for level E functionality"
    )
    config.addinivalue_line(
        "markers", "level_f: marks tests for level F functionality"
    )
    config.addinivalue_line(
        "markers", "level_g: marks tests for level G functionality"
    )
