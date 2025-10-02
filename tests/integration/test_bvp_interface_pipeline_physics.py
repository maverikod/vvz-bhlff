"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Physical validation tests for BVP interface pipeline.

This module provides comprehensive integration tests for the BVP interface
pipeline, ensuring physical consistency and theoretical correctness
of the BVP interface components.

Physical Meaning:
    Tests validate the BVP interface pipeline:
    - Interface coordination of all BVP components
    - Physical consistency maintenance
    - Proper data flow between components

Mathematical Foundation:
    Tests interface coordination of:
    - Envelope solver
    - Postulate validation
    - Quench detection
    - Impedance calculation

Example:
    >>> pytest tests/integration/test_bvp_interface_pipeline_physics.py -v
"""

import pytest
import numpy as np
from typing import Dict, Any

from bhlff.core.domain import Domain
from bhlff.core.bvp.interface.interface_facade import BVPInterface
from bhlff.core.bvp.constants.bvp_constants_advanced import BVPConstantsAdvanced


class TestBVPInterfacePipelinePhysics:
    """BVP interface pipeline physical validation tests."""

    @pytest.fixture
    def domain_7d(self):
        """Create 7D domain for complete pipeline testing."""
        return Domain(
            L=2.0,  # Larger domain for better physics
            N=64,   # Higher resolution
            dimensions=3,
            N_phi=32,  # More phase points
            N_t=128,   # More time points
            T=2.0      # Longer evolution
        )

    @pytest.fixture
    def bvp_constants(self):
        """Create BVP constants for complete pipeline testing."""
        config = {
            "envelope_equation": {
                "kappa_0": 1.0,
                "kappa_2": 0.1,
                "chi_prime": 1.0,
                "chi_double_prime_0": 0.01,
                "k0_squared": 4.0,
            },
            "basic_material": {
                "mu": 1.0,
                "beta": 1.5,
                "lambda_param": 0.1,
            }
        }
        return BVPConstantsAdvanced(config)

    @pytest.fixture
    def bvp_interface(self, domain_7d, bvp_constants):
        """Create BVP interface for complete pipeline testing."""
        return BVPInterface(domain_7d, bvp_constants)

    def test_bvp_interface_physics(self, domain_7d, bvp_interface):
        """
        Test BVP interface physics.
        
        Physical Meaning:
            Validates that the BVP interface correctly coordinates
            all BVP components and maintains physical consistency.
            
        Mathematical Foundation:
            Tests interface coordination of:
            - Envelope solver
            - Postulate validation
            - Quench detection
            - Impedance calculation
        """
        # Create test source
        source = self._generate_physical_source(domain_7d)
        
        # Test interface operations
        interface_results = bvp_interface.process_source(source)
        
        # Physical validation 1: Interface should return valid results
        assert 'envelope' in interface_results, "Interface missing envelope solution"
        assert 'postulates' in interface_results, "Interface missing postulate results"
        assert 'quenches' in interface_results, "Interface missing quench results"
        assert 'impedance' in interface_results, "Interface missing impedance results"
        
        # Physical validation 2: All results should be physically meaningful
        envelope = interface_results['envelope']
        assert np.all(np.isfinite(envelope)), "Interface envelope contains non-finite values"
        
        postulates = interface_results['postulates']
        assert isinstance(postulates, dict), "Interface postulates not a dictionary"
        
        quenches = interface_results['quenches']
        assert np.all((quenches == 0) | (quenches == 1)), "Interface quenches not binary"
        
        impedance = interface_results['impedance']
        assert np.all(np.isfinite(impedance)), "Interface impedance contains non-finite values"

    def _generate_physical_source(self, domain: Domain) -> np.ndarray:
        """Generate a physical source for testing."""
        source = np.zeros(domain.shape)
        
        # Create localized source in center
        center = domain.N // 2
        source[center-2:center+3, center-2:center+3, center-2:center+3, 
               :, :, :, :] = 1.0
        
        return source
