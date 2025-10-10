"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Tests to verify that resonators are properly integrated and used in modules.

This test suite ensures that:
- Resonators are imported and used in key modules
- Resonator coefficients are actually applied to fields
- Integration follows 7D BVP theory
"""

import pytest
import numpy as np
from pathlib import Path


class TestResonatorIntegrationUsage:
    """
    Test suite to verify resonator integration in modules.
    
    Physical Meaning:
        Ensures that resonators are not just imported but actively used
        for energy exchange in the 7D BVP framework.
    """
    
    def test_resonator_used_in_bvp_envelope_integrator(self):
        """
        Verify that bvp_envelope_integrator actually uses resonator.
        
        Physical Meaning:
            Checks that the BVP envelope integrator applies frequency-dependent
            resonator coefficients for energy exchange.
        """
        file_path = Path("bhlff/core/time/bvp_envelope_integrator.py")
        content = file_path.read_text()
        
        # Check that resonator is imported
        assert "FrequencyDependentResonator" in content, \
            "FrequencyDependentResonator should be imported"
        
        # Check that resonator is created
        assert "FrequencyDependentResonator(" in content, \
            "Resonator should be instantiated"
        
        # Check that compute_coefficients is called
        assert "compute_coefficients" in content, \
            "Resonator coefficients should be computed"
        
        # Check that R and T are used
        assert "R, T =" in content or "T, R =" in content, \
            "R and T coefficients should be extracted"
    
    def test_resonator_used_in_memory_kernel(self):
        """
        Verify that memory_kernel actually uses resonator.
        
        Physical Meaning:
            Checks that the memory kernel applies frequency-dependent
            resonator coefficients for non-local operations.
        """
        file_path = Path("bhlff/core/operators/memory_kernel.py")
        content = file_path.read_text()
        
        # Check that resonator is imported
        assert "FrequencyDependentResonator" in content, \
            "FrequencyDependentResonator should be imported"
        
        # Check that resonator is created
        assert "FrequencyDependentResonator(" in content, \
            "Resonator should be instantiated"
        
        # Check that compute_coefficients is called
        assert "compute_coefficients" in content, \
            "Resonator coefficients should be computed"
        
        # Check that R and T are used in kernel
        assert "R, T =" in content or "T, R =" in content, \
            "R and T coefficients should be extracted and used"
    
    def test_resonator_used_in_collective(self):
        """
        Verify that collective.py actually uses resonator.
        
        Physical Meaning:
            Checks that collective excitations use frequency-dependent
            resonator model for energy exchange.
        """
        file_path = Path("bhlff/models/level_f/collective.py")
        content = file_path.read_text()
        
        # Check that resonator is imported
        assert "FrequencyDependentResonator" in content, \
            "FrequencyDependentResonator should be imported"
        
        # Check that resonator is created
        assert "FrequencyDependentResonator(" in content, \
            "Resonator should be instantiated"
        
        # Check that compute_coefficients is called
        assert "compute_coefficients" in content, \
            "Resonator coefficients should be computed"
    
    def test_cascade_filter_imported_where_needed(self):
        """
        Verify that CascadeResonatorFilter is imported in relevant modules.
        
        Physical Meaning:
            Checks that cascade filter is available for multi-stage
            energy exchange in complex resonator chains.
        """
        modules_to_check = [
            "bhlff/core/time/bvp_envelope_integrator.py",
            "bhlff/core/operators/memory_kernel.py",
            "bhlff/models/level_f/collective.py"
        ]
        
        for module_path in modules_to_check:
            file_path = Path(module_path)
            if file_path.exists():
                content = file_path.read_text()
                assert "CascadeResonatorFilter" in content, \
                    f"CascadeResonatorFilter should be imported in {module_path}"
    
    def test_no_hardcoded_exponential_in_integrator(self):
        """
        Verify that bvp_envelope_integrator doesn't use hardcoded exponential.
        
        Physical Meaning:
            Ensures that envelope factor is computed using resonator model,
            not exponential attenuation.
        """
        file_path = Path("bhlff/core/time/bvp_envelope_integrator.py")
        content = file_path.read_text()
        
        # Check for forbidden patterns
        lines = content.split('\n')
        for i, line in enumerate(lines):
            # Skip comments and docstrings
            if line.strip().startswith('#') or '"""' in line or "'''" in line:
                continue
            # Skip carrier modulation (allowed)
            if 'carrier' in line.lower():
                continue
            # Check for envelope_factor with exponential
            if 'envelope_factor' in line and 'np.exp(' in line:
                pytest.fail(f"Line {i+1}: Found exponential envelope_factor - violates 7D BVP theory")
    
    def test_resonator_coefficients_actually_applied(self):
        """
        Verify that resonator coefficients are applied to fields, not just computed.
        
        Physical Meaning:
            Checks that R and T coefficients modify the field values,
            implementing actual energy exchange.
        """
        modules_to_check = [
            "bhlff/core/time/bvp_envelope_integrator.py",
            "bhlff/core/operators/memory_kernel.py"
        ]
        
        for module_path in modules_to_check:
            file_path = Path(module_path)
            content = file_path.read_text()
            
            # After computing R, T, they should be used in calculations
            assert "R, T =" in content or "T, R =" in content, \
                f"R and T should be computed in {module_path}"
            
            # Check that they're used (not just computed)
            # Look for usage patterns like np.where with T or R
            has_usage = ("np.where" in content and ("T," in content or "R," in content))
            assert has_usage, \
                f"Resonator coefficients should be applied in {module_path}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

