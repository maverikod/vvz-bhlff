"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Simple test for BVP Impedance Calculator.

This module provides a simple test for BVP impedance calculator
without the full BVPCore to avoid memory issues.
"""

import pytest
import numpy as np
from typing import Dict, Any

from bhlff.core.domain import Domain
from bhlff.core.bvp.bvp_impedance_calculator import BVPImpedanceCalculator
from bhlff.core.bvp.bvp_constants import BVPConstants


class TestBVPImpedanceCalculatorSimple:
    """
    Simple test for BVP Impedance Calculator.
    """

    @pytest.fixture
    def domain_7d(self):
        """Create 7D domain for testing."""
        return Domain(
            L=1.0,
            N=4,  # Minimal resolution for memory
            dimensions=7,
            N_phi=4,  # Minimal phase resolution
            N_t=8,  # Minimal temporal resolution
            T=1.0,
        )

    @pytest.fixture
    def config(self):
        """Create configuration for impedance calculator."""
        return {
            "frequency_range": [1e15, 1e20],
            "frequency_points": 100,
            "boundary_conditions": "periodic",
            "quality_factor_threshold": 0.1,
        }

    @pytest.fixture
    def constants(self, config):
        """Create BVP constants."""
        return BVPConstants(config)

    def test_impedance_calculator_creation(self, domain_7d, config, constants):
        """Test impedance calculator creation."""
        calculator = BVPImpedanceCalculator(domain_7d, config, constants)
        
        assert calculator.domain == domain_7d
        assert calculator.config == config
        assert calculator.constants == constants

    def test_impedance_calculator_parameters(self, domain_7d, config, constants):
        """Test impedance calculator parameters."""
        calculator = BVPImpedanceCalculator(domain_7d, config, constants)
        parameters = calculator.get_parameters()
        
        assert "frequency_range" in parameters
        assert "frequency_points" in parameters
        assert "boundary_conditions" in parameters
        assert "quality_factor_threshold" in parameters

    def test_impedance_calculation_simple(self, domain_7d, config, constants):
        """Test simple impedance calculation."""
        calculator = BVPImpedanceCalculator(domain_7d, config, constants)
        
        # Create simple envelope
        envelope = np.ones(domain_7d.shape, dtype=complex)
        
        # Calculate impedance
        impedance = calculator.compute_impedance(envelope)
        
        # Check results
        assert isinstance(impedance, dict)
        assert "admittance" in impedance
        assert "reflection" in impedance
        assert "transmission" in impedance
        assert "peaks" in impedance
        
        # Check admittance
        admittance = impedance["admittance"]
        assert isinstance(admittance, np.ndarray)
        assert len(admittance) > 0  # Should have some frequency points
        assert np.all(np.isfinite(admittance))
        
        # Check reflection
        reflection = impedance["reflection"]
        assert isinstance(reflection, np.ndarray)
        assert len(reflection) > 0  # Should have some frequency points
        assert np.all(np.isfinite(reflection))
        
        # Check transmission
        transmission = impedance["transmission"]
        assert isinstance(transmission, np.ndarray)
        assert len(transmission) > 0  # Should have some frequency points
        assert np.all(np.isfinite(transmission))
        
        # Check peaks
        peaks = impedance["peaks"]
        assert isinstance(peaks, dict)
        assert "frequencies" in peaks
        assert "quality_factors" in peaks

    def test_impedance_calculator_components(self, domain_7d, config, constants):
        """Test impedance calculator components."""
        calculator = BVPImpedanceCalculator(domain_7d, config, constants)
        
        # Test impedance core
        impedance_core = calculator.get_impedance_core()
        assert impedance_core is not None
        
        # Test resonance detector
        resonance_detector = calculator.get_resonance_detector()
        assert resonance_detector is not None

    def test_quality_factor_threshold(self, domain_7d, config, constants):
        """Test quality factor threshold setting."""
        calculator = BVPImpedanceCalculator(domain_7d, config, constants)
        
        # Set new threshold
        new_threshold = 0.5
        calculator.set_quality_factor_threshold(new_threshold)
        
        # Check if threshold was set
        detector = calculator.get_resonance_detector()
        assert detector.get_quality_factor_threshold() == new_threshold


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
