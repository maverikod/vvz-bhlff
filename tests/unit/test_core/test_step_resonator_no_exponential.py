"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Tests to verify that step resonator implementation contains no exponential damping.

This test suite ensures compliance with 7D BVP theory:
- No exponential attenuation functions
- Only step functions with T/R coefficients
- Frequency-dependent coefficients use step model
"""

import pytest
import numpy as np
from pathlib import Path


class TestStepResonatorNoExponential:
    """
    Test suite to verify absence of exponential functions in step resonator.
    
    Physical Meaning:
        Ensures that the step resonator implementation follows 7D BVP theory
        by using step functions instead of exponential attenuation.
    """
    
    def test_no_exponential_in_step_resonator_code(self):
        """
        Verify that step_resonator.py contains no exponential functions.
        
        Physical Meaning:
            Checks that the implementation uses step functions (np.where)
            instead of exponential attenuation (np.exp).
        """
        file_path = Path("bhlff/core/bvp/boundary/step_resonator.py")
        assert file_path.exists(), f"File {file_path} not found"
        
        content = file_path.read_text()
        
        # Check for forbidden exponential patterns
        assert "np.exp(" not in content, \
            "Found np.exp() in step_resonator.py - violates 7D BVP theory"
        assert "math.exp(" not in content, \
            "Found math.exp() in step_resonator.py - violates 7D BVP theory"
        
        # Verify step functions are used
        assert "np.where(" in content, \
            "Step functions (np.where) should be used for frequency-dependent coefficients"
    
    def test_no_exponential_in_step_resonator_docstring(self):
        """
        Verify that step_resonator.py documentation doesn't mention exponential attenuation.
        
        Physical Meaning:
            Ensures that the documentation accurately describes the step resonator
            model without referencing classical exponential damping.
        """
        file_path = Path("bhlff/core/bvp/boundary/step_resonator.py")
        content = file_path.read_text()
        
        # Check that documentation mentions step functions, not exponential
        assert "step function" in content.lower() or "step resonator" in content.lower(), \
            "Documentation should mention step functions"
        assert "no exponential" in content.lower(), \
            "Documentation should explicitly state 'no exponential'"
    
    def test_frequency_dependent_resonator_uses_step_functions(self):
        """
        Verify that FrequencyDependentResonator uses step functions.
        
        Physical Meaning:
            Ensures that frequency-dependent coefficients are computed
            using step functions (np.where) instead of exponential decay.
        """
        from bhlff.core.bvp.boundary.step_resonator import FrequencyDependentResonator
        
        resonator = FrequencyDependentResonator(R0=0.1, T0=0.9, omega0=1.0)
        
        # Test with frequencies below and above omega0
        frequencies = np.array([0.5, 1.5, 2.0])
        R, T = resonator.compute_coefficients(frequencies)
        
        # Below omega0: should have R0 and T0
        assert R[0] == pytest.approx(0.1), \
            "R coefficient for frequency < omega0 should equal R0"
        assert T[0] == pytest.approx(0.9), \
            "T coefficient for frequency < omega0 should equal T0"
        
        # Above omega0: should be zero (step function)
        assert R[1] == pytest.approx(0.0), \
            "R coefficient for frequency >= omega0 should be 0 (step function)"
        assert T[1] == pytest.approx(0.0), \
            "T coefficient for frequency >= omega0 should be 0 (step function)"
        assert R[2] == pytest.approx(0.0), \
            "R coefficient for frequency >= omega0 should be 0 (step function)"
        assert T[2] == pytest.approx(0.0), \
            "T coefficient for frequency >= omega0 should be 0 (step function)"
    
    def test_cascade_filter_uses_step_functions(self):
        """
        Verify that CascadeResonatorFilter uses step functions.
        
        Physical Meaning:
            Ensures that cascade filter applies step resonators
            without exponential attenuation.
        """
        from bhlff.core.bvp.boundary.step_resonator import CascadeResonatorFilter
        
        cascade = CascadeResonatorFilter(stages=3, base_R=0.1, base_T=0.9)
        
        # Verify all resonators in cascade are FrequencyDependentResonator
        assert len(cascade.resonators) == 3, "Cascade should have 3 stages"
        
        # Test cascade filter application
        field = np.random.randn(10, 10, 10) + 1j * np.random.randn(10, 10, 10)
        frequencies = np.linspace(0, 2.0, 10)
        
        result = cascade.apply_cascade_filter(field, frequencies)
        
        # Result should be a valid field
        assert result.shape == field.shape, "Cascade filter should preserve shape"
        assert np.all(np.isfinite(result)), "Cascade filter should produce finite values"
    
    def test_resonator_coefficients_are_step_functions(self):
        """
        Verify that resonator coefficients follow step function behavior.
        
        Physical Meaning:
            Checks that R(ω) and T(ω) are discontinuous step functions,
            not smooth exponential functions.
        """
        from bhlff.core.bvp.boundary.step_resonator import FrequencyDependentResonator
        
        resonator = FrequencyDependentResonator(R0=0.2, T0=0.8, omega0=1.0)
        
        # Test frequencies around cutoff
        frequencies = np.array([0.9, 0.99, 0.999, 1.0, 1.001, 1.01, 1.1])
        R, T = resonator.compute_coefficients(frequencies)
        
        # Below omega0: constant values
        assert R[0] == R[1] == R[2] == 0.2, \
            "R should be constant (R0) for frequencies < omega0"
        assert T[0] == T[1] == T[2] == 0.8, \
            "T should be constant (T0) for frequencies < omega0"
        
        # At and above omega0: zero (step discontinuity)
        assert R[3] == R[4] == R[5] == R[6] == 0.0, \
            "R should be 0 for frequencies >= omega0 (step function)"
        assert T[3] == T[4] == T[5] == T[6] == 0.0, \
            "T should be 0 for frequencies >= omega0 (step function)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

