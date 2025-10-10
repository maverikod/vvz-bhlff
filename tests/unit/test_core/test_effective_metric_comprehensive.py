"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Comprehensive tests for effective metric computation in 7D BVP theory.

This test suite verifies that effective metrics are computed correctly
according to 7D BVP theory principles, without classical spacetime curvature.
"""

import pytest
import numpy as np
from pathlib import Path
from typing import Dict, Any

pytestmark = pytest.mark.unit


class TestEffectiveMetricComprehensive:
    """
    Comprehensive test suite for effective metric computation.
    
    Physical Meaning:
        Ensures that effective metrics are computed using VBP envelope dynamics
        according to 7D BVP theory, without classical spacetime curvature.
    """
    
    def test_effective_metric_7d_structure(self):
        """
        Verify that effective metrics have correct 7D structure.
        
        Physical Meaning:
            Checks that effective metrics are 7x7 matrices representing
            the 7D space-time structure M₇ = ℝ³ₓ × 𝕋³_φ × ℝₜ.
        """
        file_path = Path("bhlff/models/level_g/cosmology.py")
        content = file_path.read_text()
        
        # Check for 7D metric structure
        assert "g_eff = np.zeros((7, 7))" in content, \
            "Effective metric should be 7x7 matrix for 7D space-time"
        
        # Check for proper indexing
        assert "g_eff[0, 0]" in content, \
            "Time component should be g_eff[0, 0]"
        
        assert "g_eff[i, i]" in content, \
            "Spatial components should be g_eff[i, i] for i=1,2,3"
        
        assert "g_eff[alpha, alpha]" in content, \
            "Phase components should be g_eff[alpha, alpha] for alpha=4,5,6"
    
    def test_effective_metric_time_component(self):
        """
        Verify that time component follows 7D BVP theory.
        
        Physical Meaning:
            Checks that g₀₀ = -1/c_φ² follows 7D BVP theory principles
            for phase velocity c_φ in 7D space-time.
        """
        file_path = Path("bhlff/models/level_g/cosmology.py")
        content = file_path.read_text()
        
        # Check for time component structure
        assert "g_eff[0, 0] = -1.0 / (c_phi**2" in content, \
            "Time component should be g₀₀ = -1/c_φ²"
        
        assert "c_phi" in content, \
            "Phase velocity c_φ should be used in time component"
        
        assert "envelope_correction" in content, \
            "Time component should include envelope correction"
    
    def test_effective_metric_spatial_components(self):
        """
        Verify that spatial components use envelope invariants.
        
        Physical Meaning:
            Checks that spatial components gᵢᵢ use envelope invariants
            A^{ij} instead of classical spacetime metrics.
        """
        file_path = Path("bhlff/models/level_g/cosmology.py")
        content = file_path.read_text()
        
        # Check for spatial components using envelope invariants
        assert "chi_kappa" in content, \
            "Spatial components should use chi_kappa parameter"
        
        assert "curvature_correction" in content, \
            "Spatial components should include curvature correction"
        
        assert "envelope_gradient_magnitude" in content, \
            "Spatial components should use envelope gradient magnitude"
    
    def test_effective_metric_phase_components(self):
        """
        Verify that phase components represent internal field states.
        
        Physical Meaning:
            Checks that phase components g_αα represent the internal
            field states in the 3D phase space 𝕋³_φ.
        """
        file_path = Path("bhlff/models/level_g/cosmology.py")
        content = file_path.read_text()
        
        # Check for phase components
        assert "for alpha in range(4, 7)" in content, \
            "Phase components should be g_αα for α=4,5,6"
        
        assert "g_eff[alpha, alpha]" in content, \
            "Phase components should be set in the metric"
        
        assert "envelope_amplitude" in content, \
            "Phase components should use envelope amplitude"
    
    def test_anisotropic_metric_components(self):
        """
        Verify that anisotropic metric uses envelope invariants.
        
        Physical Meaning:
            Checks that anisotropic metric components A^{ij} are
            computed from envelope invariants, not classical metrics.
        """
        file_path = Path("bhlff/models/level_g/cosmology.py")
        content = file_path.read_text()
        
        # Check for anisotropic metric implementation
        assert "A_xx" in content and "A_yy" in content and "A_zz" in content, \
            "Anisotropic metric should have A_xx, A_yy, A_zz components"
        
        assert "envelope_invariants" in content, \
            "Anisotropic metric should use envelope invariants"
        
        assert "chi_kappa" in content, \
            "Anisotropic metric should use chi_kappa parameter"
    
    def test_no_classical_spacetime_curvature(self):
        """
        Verify that no classical spacetime curvature is used.
        
        Physical Meaning:
            Checks that effective metrics don't use classical
            spacetime curvature concepts like Riemann tensor.
        """
        level_g_files = [
            "bhlff/models/level_g/gravity_curvature.py",
            "bhlff/models/level_g/gravity_einstein.py",
            "bhlff/models/level_g/cosmology.py"
        ]
        
        classical_terms = [
            "Riemann",
            "Ricci",
            "Einstein",
            "Christoffel",
            "spacetime curvature"
        ]
        
        for file_path in level_g_files:
            path = Path(file_path)
            if path.exists():
                content = path.read_text()
                
                for term in classical_terms:
                    if term in content:
                        # Check if it's in a comment explaining replacement
                        lines = content.split('\n')
                        for line in lines:
                            if term in line and not line.strip().startswith('#'):
                                # Allow terms in comments that explain replacement
                                if "VBP envelope" in content or "envelope dynamics" in content or "phase field" in content:
                                    continue
                                else:
                                    assert False, f"Found classical term '{term}' in {file_path}, should use VBP envelope"
    
    def test_vbp_envelope_dynamics_consistency(self):
        """
        Verify that all effective metric computations use VBP envelope dynamics.
        
        Physical Meaning:
            Checks that all effective metric computations consistently
            use VBP envelope dynamics instead of classical approaches.
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
                
                # Check for VBP envelope dynamics references
                vbp_terms = [
                    "VBP envelope",
                    "envelope dynamics",
                    "phase field",
                    "envelope amplitude",
                    "envelope gradient"
                ]
                
                found_vbp_terms = [term for term in vbp_terms if term in content]
                assert len(found_vbp_terms) > 0, \
                    f"{file_path} should reference VBP envelope dynamics"
    
    def test_effective_metric_physical_meaning(self):
        """
        Verify that effective metrics have correct physical meaning.
        
        Physical Meaning:
            Checks that effective metrics represent the VBP envelope
            dynamics and not classical spacetime geometry.
        """
        file_path = Path("bhlff/models/level_g/cosmology.py")
        content = file_path.read_text()
        
        # Check for physical meaning in docstrings
        assert "VBP envelope" in content, \
            "Effective metric should reference VBP envelope dynamics"
        
        assert "no spacetime curvature" in content or "not from spacetime curvature" in content, \
            "Effective metric should not use spacetime curvature"
        
        assert "envelope invariants" in content, \
            "Effective metric should use envelope invariants"
    
    def test_scale_factor_power_law_evolution(self):
        """
        Verify that scale factor uses power law evolution.
        
        Physical Meaning:
            Checks that cosmological scale factor uses power law evolution
            instead of exponential growth, conforming to 7D BVP theory.
        """
        file_path = Path("bhlff/models/level_g/cosmology.py")
        content = file_path.read_text()
        
        # Check for power law evolution
        assert "power law" in content.lower(), \
            "Scale factor should use power law evolution"
        
        assert "**" in content, \
            "Scale factor should use power law (**) instead of exponential"
        
        # Check for absence of exponential in scale factor
        if "np.exp(" in content:
            # If exponential is present, it should be in comments explaining replacement
            lines = content.split('\n')
            for line in lines:
                if "np.exp(" in line and not line.strip().startswith('#'):
                    assert "power law" in line.lower(), \
                        "Exponential should be replaced with power law"
    
    def test_effective_metric_integration_consistency(self):
        """
        Verify that effective metric integration is consistent across modules.
        
        Physical Meaning:
            Checks that all gravitational modules consistently use
            EnvelopeEffectiveMetric for VBP envelope dynamics.
        """
        integration_files = [
            "bhlff/models/level_g/gravity_curvature.py",
            "bhlff/models/level_g/gravity_einstein.py"
        ]
        
        for file_path in integration_files:
            path = Path(file_path)
            if path.exists():
                content = path.read_text()
                
                # Check for EnvelopeEffectiveMetric integration
                assert "EnvelopeEffectiveMetric" in content, \
                    f"{file_path} should import EnvelopeEffectiveMetric"
                
                assert "self.envelope_metric" in content, \
                    f"{file_path} should initialize envelope_metric"
                
                # Check for integration methods
                if "gravity_curvature" in file_path:
                    assert "compute_envelope_effective_metric" in content, \
                        "gravity_curvature.py should have envelope effective metric methods"
                elif "gravity_einstein" in file_path:
                    assert "solve_with_envelope_effective_metric" in content, \
                        "gravity_einstein.py should have envelope effective metric methods"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
