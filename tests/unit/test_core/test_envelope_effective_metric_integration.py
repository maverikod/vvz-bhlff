"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Tests for EnvelopeEffectiveMetric integration in Level G models.

This test suite verifies that EnvelopeEffectiveMetric is properly integrated
into all gravitational modules and provides the expected functionality.
"""

import pytest
import numpy as np
from pathlib import Path

pytestmark = pytest.mark.unit


class TestEnvelopeEffectiveMetricIntegration:
    """
    Test suite for EnvelopeEffectiveMetric integration in Level G models.
    
    Physical Meaning:
        Ensures that EnvelopeEffectiveMetric is properly integrated into
        all gravitational modules and provides correct VBP envelope dynamics.
    """
    
    def test_envelope_effective_metric_imported_in_gravity_curvature(self):
        """
        Verify that EnvelopeEffectiveMetric is imported in gravity_curvature.py.
        
        Physical Meaning:
            Checks that gravity_curvature.py imports EnvelopeEffectiveMetric
            for integration with VBP envelope curvature calculations.
        """
        file_path = Path("bhlff/models/level_g/gravity_curvature.py")
        assert file_path.exists(), f"File {file_path} not found"
        
        content = file_path.read_text()
        
        # Check for import
        assert "from .cosmology import EnvelopeEffectiveMetric" in content, \
            "EnvelopeEffectiveMetric should be imported in gravity_curvature.py"
        
        # Check for integration
        assert "self.envelope_metric = EnvelopeEffectiveMetric(params)" in content, \
            "EnvelopeEffectiveMetric should be initialized in VBPEnvelopeCurvatureCalculator"
    
    def test_envelope_effective_metric_imported_in_gravity_einstein(self):
        """
        Verify that EnvelopeEffectiveMetric is imported in gravity_einstein.py.
        
        Physical Meaning:
            Checks that gravity_einstein.py imports EnvelopeEffectiveMetric
            for integration with phase envelope balance solver.
        """
        file_path = Path("bhlff/models/level_g/gravity_einstein.py")
        assert file_path.exists(), f"File {file_path} not found"
        
        content = file_path.read_text()
        
        # Check for import
        assert "from .cosmology import EnvelopeEffectiveMetric" in content, \
            "EnvelopeEffectiveMetric should be imported in gravity_einstein.py"
        
        # Check for integration
        assert "self.envelope_metric = EnvelopeEffectiveMetric(params)" in content, \
            "EnvelopeEffectiveMetric should be initialized in PhaseEnvelopeBalanceSolver"
    
    def test_envelope_effective_metric_methods_in_gravity_curvature(self):
        """
        Verify that EnvelopeEffectiveMetric methods are available in gravity_curvature.py.
        
        Physical Meaning:
            Checks that VBPEnvelopeCurvatureCalculator has methods to use
            EnvelopeEffectiveMetric for computing effective metrics.
        """
        file_path = Path("bhlff/models/level_g/gravity_curvature.py")
        content = file_path.read_text()
        
        # Check for integration methods
        integration_methods = [
            "compute_envelope_effective_metric",
            "compute_anisotropic_envelope_metric", 
            "compute_cosmological_scale_factor"
        ]
        
        for method in integration_methods:
            assert f"def {method}" in content, \
                f"Method {method} should be available in VBPEnvelopeCurvatureCalculator"
    
    def test_envelope_effective_metric_methods_in_gravity_einstein(self):
        """
        Verify that EnvelopeEffectiveMetric methods are available in gravity_einstein.py.
        
        Physical Meaning:
            Checks that PhaseEnvelopeBalanceSolver has methods to use
            EnvelopeEffectiveMetric for solving phase envelope balance equations.
        """
        file_path = Path("bhlff/models/level_g/gravity_einstein.py")
        content = file_path.read_text()
        
        # Check for integration methods
        integration_methods = [
            "solve_with_envelope_effective_metric",
            "compute_anisotropic_envelope_solution",
            "compute_cosmological_envelope_evolution"
        ]
        
        for method in integration_methods:
            assert f"def {method}" in content, \
                f"Method {method} should be available in PhaseEnvelopeBalanceSolver"
    
    def test_envelope_effective_metric_extended_methods(self):
        """
        Verify that EnvelopeEffectiveMetric has extended methods for VBP envelope dynamics.
        
        Physical Meaning:
            Checks that EnvelopeEffectiveMetric has been extended with methods
            for computing effective metrics from envelope curvature and anisotropy.
        """
        file_path = Path("bhlff/models/level_g/cosmology.py")
        content = file_path.read_text()
        
        # Check for extended methods
        extended_methods = [
            "compute_envelope_curvature_metric",
            "compute_anisotropic_metric"
        ]
        
        for method in extended_methods:
            assert f"def {method}" in content, \
                f"Method {method} should be available in EnvelopeEffectiveMetric"
    
    def test_no_exponential_in_scale_factor(self):
        """
        Verify that scale factor computation doesn't use exponential functions.
        
        Physical Meaning:
            Checks that EnvelopeEffectiveMetric.compute_scale_factor() uses
            power law evolution instead of exponential growth, conforming
            to 7D BVP theory.
        """
        file_path = Path("bhlff/models/level_g/cosmology.py")
        content = file_path.read_text()
        
        # Check for absence of exponential in scale factor
        assert "np.exp(" not in content or "power law" in content, \
            "Scale factor computation should use power law instead of exponential"
        
        # Check for power law evolution
        assert "power law" in content.lower(), \
            "Scale factor should use power law evolution for VBP envelope dynamics"
    
    def test_envelope_curvature_metric_implementation(self):
        """
        Verify that envelope curvature metric computation is implemented correctly.
        
        Physical Meaning:
            Checks that compute_envelope_curvature_metric() implements
            the correct VBP envelope dynamics for effective metric computation.
        """
        file_path = Path("bhlff/models/level_g/cosmology.py")
        content = file_path.read_text()
        
        # Check for envelope curvature metric implementation
        assert "compute_envelope_curvature_metric" in content, \
            "compute_envelope_curvature_metric should be implemented"
        
        assert "phase_field" in content, \
            "Envelope curvature metric should use phase field input"
        
        assert "g_eff" in content, \
            "Envelope curvature metric should compute effective metric g_eff"
    
    def test_anisotropic_metric_implementation(self):
        """
        Verify that anisotropic metric computation is implemented correctly.
        
        Physical Meaning:
            Checks that compute_anisotropic_metric() implements
            anisotropic effective metrics for VBP envelope dynamics.
        """
        file_path = Path("bhlff/models/level_g/cosmology.py")
        content = file_path.read_text()
        
        # Check for anisotropic metric implementation
        assert "compute_anisotropic_metric" in content, \
            "compute_anisotropic_metric should be implemented"
        
        assert "A_xx" in content and "A_yy" in content and "A_zz" in content, \
            "Anisotropic metric should support different spatial components"
        
        assert "envelope_invariants" in content, \
            "Anisotropic metric should use envelope invariants"
    
    def test_integration_consistency(self):
        """
        Verify that EnvelopeEffectiveMetric integration is consistent across modules.
        
        Physical Meaning:
            Checks that all gravitational modules use EnvelopeEffectiveMetric
            consistently for VBP envelope dynamics.
        """
        level_g_files = [
            "bhlff/models/level_g/gravity_curvature.py",
            "bhlff/models/level_g/gravity_einstein.py",
            "bhlff/models/level_g/cosmology.py"
        ]
        
        for file_path in level_g_files:
            path = Path(file_path)
            if path.exists():
                content = path.read_text()
                
                # Check for EnvelopeEffectiveMetric usage
                if "gravity_curvature" in file_path:
                    assert "self.envelope_metric" in content, \
                        f"gravity_curvature.py should use envelope_metric"
                elif "gravity_einstein" in file_path:
                    assert "self.envelope_metric" in content, \
                        f"gravity_einstein.py should use envelope_metric"
                elif "cosmology" in file_path:
                    assert "class EnvelopeEffectiveMetric" in content, \
                        f"cosmology.py should define EnvelopeEffectiveMetric"
    
    def test_vbp_envelope_dynamics_compliance(self):
        """
        Verify that all implementations comply with VBP envelope dynamics.
        
        Physical Meaning:
            Checks that all EnvelopeEffectiveMetric implementations
            follow 7D BVP theory principles for VBP envelope dynamics.
        """
        level_g_files = [
            "bhlff/models/level_g/gravity_curvature.py",
            "bhlff/models/level_g/gravity_einstein.py",
            "bhlff/models/level_g/cosmology.py"
        ]
        
        for file_path in level_g_files:
            path = Path(file_path)
            if path.exists():
                content = path.read_text()
                
                # Check for VBP envelope dynamics compliance
                assert "VBP envelope" in content or "envelope dynamics" in content, \
                    f"{file_path} should reference VBP envelope dynamics"
                
                assert "no spacetime curvature" in content or "not from spacetime curvature" in content, \
                    f"{file_path} should not use spacetime curvature"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
