"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Comprehensive tests for BVP resonance detector core module.

This module provides comprehensive tests for the BVP resonance detector core module,
covering all classes and methods to achieve high test coverage.
"""

import pytest
import numpy as np
from typing import Dict, Any

from bhlff.core.domain import Domain
from bhlff.core.bvp.resonance_detector_core import ResonanceDetector
from bhlff.core.bvp.constants.bvp_constants_advanced import BVPConstantsAdvanced


class TestResonanceDetectorCoreCoverage:
    """Comprehensive tests for BVP resonance detector core module."""

    @pytest.fixture
    def domain_7d(self):
        """Create 7D domain for testing."""
        return Domain(
            L=1.0,
            N=8,
            dimensions=7,
            N_phi=4,
            N_t=8,
            T=1.0
        )

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return {
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

    @pytest.fixture
    def bvp_constants(self, config):
        """Create BVP constants."""
        return BVPConstantsAdvanced(config)

    def test_resonance_detector_core_creation(self, domain_7d, bvp_constants):
        """Test resonance detector core creation."""
        # Create a simple BVPConstants instance instead of BVPConstantsAdvanced
        from bhlff.core.bvp.bvp_constants import BVPConstants
        simple_constants = BVPConstants()
        resonance_detector_core = ResonanceDetector(simple_constants)
        
        # Test basic properties
        assert hasattr(resonance_detector_core, 'constants')
        assert hasattr(resonance_detector_core, 'quality_factor_threshold')
        assert hasattr(resonance_detector_core, 'peak_detector')
        assert hasattr(resonance_detector_core, 'quality_analyzer')
        assert resonance_detector_core.constants == simple_constants

    def test_resonance_detector_core_find_resonance_peaks(self, domain_7d, bvp_constants):
        """Test resonance detector core find resonance peaks method."""
        from bhlff.core.bvp.bvp_constants import BVPConstants
        simple_constants = BVPConstants()
        resonance_detector_core = ResonanceDetector(simple_constants)
        
        # Create test frequencies and admittance
        frequencies = np.linspace(0.1, 10.0, 100)
        admittance = np.exp(-frequencies) * np.exp(1j * frequencies)
        
        # Test find_resonance_peaks method
        peaks = resonance_detector_core.find_resonance_peaks(frequencies, admittance)
        
        # Validate result
        assert isinstance(peaks, dict)
        assert 'frequencies' in peaks
        assert 'quality_factors' in peaks

    def test_resonance_detector_core_quality_factor_threshold(self, domain_7d, bvp_constants):
        """Test resonance detector core quality factor threshold methods."""
        from bhlff.core.bvp.bvp_constants import BVPConstants
        simple_constants = BVPConstants()
        resonance_detector_core = ResonanceDetector(simple_constants)
        
        # Test get_quality_factor_threshold method
        threshold = resonance_detector_core.get_quality_factor_threshold()
        assert isinstance(threshold, float)
        assert threshold > 0
        
        # Test set_quality_factor_threshold method
        new_threshold = 5.0
        resonance_detector_core.set_quality_factor_threshold(new_threshold)
        assert resonance_detector_core.get_quality_factor_threshold() == new_threshold

    def test_resonance_detector_core_string_representation(self, domain_7d, bvp_constants):
        """Test resonance detector core string representation."""
        from bhlff.core.bvp.bvp_constants import BVPConstants
        simple_constants = BVPConstants()
        resonance_detector_core = ResonanceDetector(simple_constants)
        
        # Test string representation
        str_repr = str(resonance_detector_core)
        assert isinstance(str_repr, str)
        assert len(str_repr) > 0
        
        # Test repr
        repr_str = repr(resonance_detector_core)
        assert isinstance(repr_str, str)
        assert len(repr_str) > 0
