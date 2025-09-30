"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Pytest configuration for BHLFF physical validation tests.

This module provides pytest configuration and fixtures for comprehensive
physical validation testing of the 7D BVP theory implementation.

Physical Meaning:
    Configures pytest for testing the physical correctness of the
    7D Base High-Frequency Field theory, ensuring theoretical
    consistency and physical validity.

Mathematical Foundation:
    Sets up testing environment for validating:
    - 7D envelope equation physics
    - U(1)³ phase structure
    - Energy conservation
    - BVP postulates
    - Spectral methods
    - Material properties

Example:
    >>> pytest tests/ -v --tb=short
"""

import pytest
import numpy as np
from typing import Dict, Any, Generator

from bhlff.core.domain import Domain
from bhlff.core.bvp.constants.bvp_constants_advanced import BVPConstantsAdvanced


@pytest.fixture(scope="session")
def test_domain_7d() -> Domain:
    """
    Create 7D domain for testing.
    
    Physical Meaning:
        Provides a 7D computational domain M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ
        for testing the BVP theory implementation.
        
    Mathematical Foundation:
        Domain with periodic boundary conditions in 7D space-time:
        M₇ = [0,L)³ × [0,2π)³ × [0,T) with uniform grid spacing.
    """
    return Domain(
        L=1.0,      # Spatial domain size
        N=32,       # Spatial resolution
        dimensions=3,
        N_phi=16,   # Phase resolution
        N_t=64,     # Temporal resolution
        T=1.0       # Temporal domain size
    )


@pytest.fixture(scope="session")
def test_domain_7d_high_res() -> Domain:
    """
    Create high-resolution 7D domain for testing.
    
    Physical Meaning:
        Provides a high-resolution 7D computational domain for
        testing spectral accuracy and convergence.
    """
    return Domain(
        L=2.0,      # Larger spatial domain
        N=64,       # Higher spatial resolution
        dimensions=3,
        N_phi=32,   # Higher phase resolution
        N_t=128,    # Higher temporal resolution
        T=2.0       # Longer temporal domain
    )


@pytest.fixture(scope="session")
def test_bvp_constants() -> BVPConstantsAdvanced:
    """
    Create BVP constants for testing.
    
    Physical Meaning:
        Provides physically meaningful BVP constants for testing
        the 7D BVP theory implementation.
        
    Mathematical Foundation:
        Constants satisfying physical constraints:
        μ > 0, β ∈ (0,2), λ ≥ 0, k₀ > 0, χ₀ > 0, κ₀ > 0
    """
    return BVPConstantsAdvanced(
        mu=1.0,         # Diffusion coefficient
        beta=1.5,       # Fractional order
        lambda_param=0.1,  # Damping parameter
        k0=2.0,         # Carrier frequency
        chi0=1.0,       # Linear susceptibility
        kappa0=1.0      # Linear stiffness
    )


@pytest.fixture(scope="session")
def test_bvp_constants_extreme() -> BVPConstantsAdvanced:
    """
    Create extreme BVP constants for testing.
    
    Physical Meaning:
        Provides extreme but physically valid BVP constants for
        testing robustness and edge cases.
    """
    return BVPConstantsAdvanced(
        mu=0.1,         # Small diffusion
        beta=0.1,       # Small fractional order
        lambda_param=0.01,  # Small damping
        k0=10.0,        # High carrier frequency
        chi0=0.1,       # Small susceptibility
        kappa0=0.1      # Small stiffness
    )


@pytest.fixture
def test_source_gaussian(domain_7d: Domain) -> np.ndarray:
    """
    Create Gaussian source for testing.
    
    Physical Meaning:
        Provides a Gaussian source with known analytical properties
        for testing BVP equation solution.
    """
    source = np.zeros(domain_7d.shape)
    
    # Create Gaussian in center
    center = domain_7d.N // 2
    x = np.arange(domain_7d.N) - center
    y = np.arange(domain_7d.N) - center
    z = np.arange(domain_7d.N) - center
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    gaussian = np.exp(-(X**2 + Y**2 + Z**2) / (2 * (domain_7d.N/8)**2))
    
    source[:, :, :, :, :, :, :] = gaussian[:, :, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    
    return source


@pytest.fixture
def test_source_sinusoidal(domain_7d: Domain) -> np.ndarray:
    """
    Create sinusoidal source for testing.
    
    Physical Meaning:
        Provides a sinusoidal source with known analytical derivatives
        for testing spectral methods.
    """
    source = np.zeros(domain_7d.shape)
    
    # Create sinusoidal pattern
    x = np.linspace(0, 2*np.pi, domain_7d.N)
    y = np.linspace(0, 2*np.pi, domain_7d.N)
    z = np.linspace(0, 2*np.pi, domain_7d.N)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    source[:, :, :, :, :, :, :] = np.sin(X) * np.sin(Y) * np.sin(Z)
    
    return source


@pytest.fixture
def test_source_localized(domain_7d: Domain) -> np.ndarray:
    """
    Create localized source for testing.
    
    Physical Meaning:
        Provides a localized source for testing boundary conditions
        and field propagation.
    """
    source = np.zeros(domain_7d.shape)
    
    # Create localized source in center
    center = domain_7d.N // 2
    source[center-2:center+3, center-2:center+3, center-2:center+3, 
           :, :, :, :] = 1.0
    
    return source


@pytest.fixture
def test_envelope_analytical(domain_7d: Domain) -> np.ndarray:
    """
    Create analytical envelope for testing.
    
    Physical Meaning:
        Provides an envelope with known analytical properties
        for testing postulate validation.
    """
    envelope = np.zeros(domain_7d.shape)
    
    # Create envelope with known properties
    center = domain_7d.N // 2
    envelope[center-4:center+5, center-4:center+5, center-4:center+5,
            :, :, :, :] = 1.0
    
    # Add phase structure
    phi1 = np.linspace(0, 2*np.pi, domain_7d.N_phi)
    phi2 = np.linspace(0, 2*np.pi, domain_7d.N_phi)
    phi3 = np.linspace(0, 2*np.pi, domain_7d.N_phi)
    
    PHI1, PHI2, PHI3 = np.meshgrid(phi1, phi2, phi3, indexing='ij')
    phase_factor = np.exp(1j * (PHI1 + PHI2 + PHI3))
    
    envelope = envelope * phase_factor
    
    return envelope


@pytest.fixture
def test_frequencies() -> np.ndarray:
    """
    Create test frequencies for frequency-dependent testing.
    
    Physical Meaning:
        Provides a range of frequencies for testing frequency-dependent
        material properties and spectral methods.
    """
    return np.logspace(-2, 2, 100)  # 0.01 to 100


@pytest.fixture
def test_amplitudes() -> np.ndarray:
    """
    Create test amplitudes for nonlinear testing.
    
    Physical Meaning:
        Provides a range of amplitudes for testing nonlinear
        material properties and effects.
    """
    return np.linspace(0.1, 2.0, 50)


@pytest.fixture
def test_scales() -> np.ndarray:
    """
    Create test scales for renormalization testing.
    
    Physical Meaning:
        Provides a range of scales for testing renormalization
        group flow and scaling behavior.
    """
    return np.logspace(-2, 2, 50)  # 0.01 to 100


@pytest.fixture(autouse=True)
def setup_test_environment():
    """
    Setup test environment.
    
    Physical Meaning:
        Configures the test environment for reproducible
        physical validation testing.
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Set numpy error handling
    np.seterr(all='raise')
    
    yield
    
    # Cleanup after tests
    np.seterr(all='warn')


def pytest_configure(config):
    """
    Configure pytest for physical validation testing.
    
    Physical Meaning:
        Configures pytest with appropriate settings for
        physical validation testing of the 7D BVP theory.
    """
    # Add custom markers
    config.addinivalue_line(
        "markers", "physics: mark test as physical validation test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection for physical validation.
    
    Physical Meaning:
        Modifies test collection to prioritize physical
        validation tests and handle slow tests appropriately.
    """
    for item in items:
        # Add physics marker to all tests in test_core
        if "test_core" in str(item.fspath):
            item.add_marker(pytest.mark.physics)
        
        # Add slow marker to integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.slow)
            item.add_marker(pytest.mark.integration)


@pytest.fixture(scope="session")
def test_results_cache() -> Dict[str, Any]:
    """
    Create test results cache.
    
    Physical Meaning:
        Provides a cache for storing test results to avoid
        recomputation in long-running physical validation tests.
    """
    return {}


def pytest_runtest_setup(item):
    """
    Setup for each test run.
    
    Physical Meaning:
        Sets up each test run with appropriate configuration
        for physical validation testing.
    """
    # Skip slow tests unless explicitly requested
    if item.get_closest_marker("slow") and not item.config.getoption("--runslow"):
        pytest.skip("slow test (use --runslow to run)")


def pytest_addoption(parser):
    """
    Add command line options for physical validation testing.
    
    Physical Meaning:
        Adds command line options for controlling physical
        validation test execution.
    """
    parser.addoption(
        "--runslow", action="store_true", default=False,
        help="run slow tests"
    )
    parser.addoption(
        "--physics-only", action="store_true", default=False,
        help="run only physics validation tests"
    )
    parser.addoption(
        "--coverage-physics", action="store_true", default=False,
        help="run coverage analysis for physics tests"
    )
