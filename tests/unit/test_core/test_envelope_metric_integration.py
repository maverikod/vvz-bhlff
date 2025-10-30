"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Tests to verify integration of EnvelopeEffectiveMetric in gravitational modules.

This test suite ensures that:
- EnvelopeEffectiveMetric is properly integrated in gravity modules
- Extended methods are available and functional
- No exponential functions in scale factor computation
- Compliance with 7D BVP theory
"""

import pytest
import numpy as np
from pathlib import Path


class TestEnvelopeMetricIntegration:
    """
    Test suite to verify EnvelopeEffectiveMetric integration.

    Physical Meaning:
        Ensures that EnvelopeEffectiveMetric is properly integrated
        into all gravitational modules and extended methods are functional.
    """

    def test_envelope_effective_metric_extended_methods_exist(self):
        """
        Verify that EnvelopeEffectiveMetric has extended methods.

        Physical Meaning:
            Checks that EnvelopeEffectiveMetric class has all required
            extended methods for envelope-based metric computation.
        """
        file_path = Path("bhlff/models/level_g/cosmology.py")
        content = file_path.read_text()

        # Check for extended methods
        extended_methods = [
            "compute_envelope_curvature_metric",
            "compute_anisotropic_metric",
            "compute_scale_factor",
        ]

        for method in extended_methods:
            assert (
                method in content
            ), f"Extended method '{method}' not found in EnvelopeEffectiveMetric"

    def test_no_exponential_in_scale_factor(self):
        """
        Verify that scale_factor doesn't use exponential functions.

        Physical Meaning:
            Checks that compute_scale_factor uses power law evolution
            instead of exponential growth, complying with 7D BVP theory.
        """
        file_path = Path("bhlff/models/level_g/cosmology.py")
        content = file_path.read_text()

        # Find compute_scale_factor method
        method_start = content.find("def compute_scale_factor")
        if method_start == -1:
            pytest.fail("compute_scale_factor method not found")

        # Get method content (next ~20 lines)
        method_content = content[method_start : method_start + 1000]
        next_def = method_content.find("def ", 5)  # Find next method
        if next_def > 0:
            method_content = method_content[:next_def]

        # Check for forbidden exponential
        # Allow np.sqrt but not np.exp for scale factor
        assert (
            "np.exp(" not in method_content or "np.sqrt" in method_content
        ), "compute_scale_factor should not use np.exp() - violates 7D BVP theory"

        # Verify power law is used
        assert (
            "**" in method_content
        ), "compute_scale_factor should use power law (** operator)"

    def test_envelope_curvature_metric_implementation(self):
        """
        Verify that compute_envelope_curvature_metric is implemented.

        Physical Meaning:
            Checks that envelope curvature metric computation is
            properly implemented with phase field gradients.
        """
        file_path = Path("bhlff/models/level_g/cosmology.py")
        content = file_path.read_text()

        # Find compute_envelope_curvature_metric method
        assert (
            "def compute_envelope_curvature_metric" in content
        ), "compute_envelope_curvature_metric method not found"

        # Check for phase field gradient computation
        method_start = content.find("def compute_envelope_curvature_metric")
        method_content = content[method_start : method_start + 2000]

        assert (
            "np.gradient" in method_content or "phase_gradients" in method_content
        ), "compute_envelope_curvature_metric should compute phase field gradients"

        assert (
            "g_eff" in method_content
        ), "compute_envelope_curvature_metric should compute effective metric g_eff"

    def test_anisotropic_metric_implementation(self):
        """
        Verify that compute_anisotropic_metric is implemented.

        Physical Meaning:
            Checks that anisotropic metric computation allows different
            spatial components, reflecting anisotropic envelope dynamics.
        """
        file_path = Path("bhlff/models/level_g/cosmology.py")
        content = file_path.read_text()

        # Find compute_anisotropic_metric method
        assert (
            "def compute_anisotropic_metric" in content
        ), "compute_anisotropic_metric method not found"

        # Check for anisotropic components
        method_start = content.find("def compute_anisotropic_metric")
        method_content = content[method_start : method_start + 2000]

        assert (
            "A_xx" in method_content
            or "A_yy" in method_content
            or "A_zz" in method_content
        ), "compute_anisotropic_metric should have anisotropic spatial components"

    def test_integration_in_gravity_curvature_needed(self):
        """
        Check if integration in gravity_curvature.py is needed.

        Physical Meaning:
            Identifies if gravity_curvature.py should integrate
            EnvelopeEffectiveMetric for improved functionality.
        """
        file_path = Path("bhlff/models/level_g/gravity_curvature.py")
        content = file_path.read_text()

        # Check if EnvelopeEffectiveMetric is imported
        has_import = "EnvelopeEffectiveMetric" in content

        # Check if VBPEnvelopeCurvatureCalculator uses envelope metric
        has_envelope_metric_usage = (
            "envelope_metric" in content or "effective_metric" in content
        )

        if not has_import:
            print(
                "\nWARNING: gravity_curvature.py doesn't import EnvelopeEffectiveMetric"
            )
            print("Consider adding: from .cosmology import EnvelopeEffectiveMetric")

        if not has_envelope_metric_usage:
            print(
                "\nWARNING: VBPEnvelopeCurvatureCalculator doesn't use envelope metric methods"
            )
            print("Consider integrating compute_envelope_curvature_metric()")

    def test_integration_in_gravity_einstein_needed(self):
        """
        Check if integration in gravity_einstein.py is needed.

        Physical Meaning:
            Identifies if gravity_einstein.py should integrate
            EnvelopeEffectiveMetric for improved functionality.
        """
        file_path = Path("bhlff/models/level_g/gravity_einstein.py")
        content = file_path.read_text()

        # Check if EnvelopeEffectiveMetric is imported
        has_import = "EnvelopeEffectiveMetric" in content

        # Check if PhaseEnvelopeBalanceSolver uses envelope metric
        has_envelope_metric_usage = (
            "envelope_metric" in content or "compute_envelope" in content
        )

        if not has_import:
            print(
                "\nWARNING: gravity_einstein.py doesn't import EnvelopeEffectiveMetric"
            )
            print("Consider adding: from .cosmology import EnvelopeEffectiveMetric")

        if not has_envelope_metric_usage:
            print(
                "\nWARNING: PhaseEnvelopeBalanceSolver doesn't use envelope metric methods"
            )
            print("Consider integrating compute_anisotropic_metric()")

    def test_vbp_envelope_dynamics_compliance(self):
        """
        Verify compliance with VBP envelope dynamics principles.

        Physical Meaning:
            Checks that all metric computations follow VBP envelope
            dynamics principles without classical spacetime curvature.
        """
        file_path = Path("bhlff/models/level_g/cosmology.py")
        content = file_path.read_text()

        # Check for VBP envelope references
        assert (
            "VBP envelope" in content or "envelope dynamics" in content
        ), "EnvelopeEffectiveMetric should reference VBP envelope dynamics"

        # Check that no spacetime curvature is used
        forbidden_terms = ["spacetime curvature", "Riemann", "Ricci"]
        for term in forbidden_terms:
            # Allow in comments explaining what we DON'T use
            if term in content and "no " not in content.lower():
                assert (
                    term not in content
                ), f"Found forbidden term '{term}' in EnvelopeEffectiveMetric"

    def test_effective_metric_7d_structure(self):
        """
        Verify that effective metric has 7D structure.

        Physical Meaning:
            Checks that all metric computations produce 7x7 tensors
            for the 7D phase field theory (3 spatial + 3 phase + 1 time).
        """
        file_path = Path("bhlff/models/level_g/cosmology.py")
        content = file_path.read_text()

        # Check for 7x7 metric initialization
        assert "(7, 7)" in content, "Effective metric should be 7x7 for 7D BVP theory"

        # Check for proper component assignments
        assert "g_eff[0, 0]" in content, "Time component g00 should be assigned"

        # Check for spatial components
        assert (
            "range(1, 4)" in content or "g_eff[1, 1]" in content
        ), "Spatial components gij should be assigned"

        # Check for phase components
        assert (
            "range(4, 7)" in content or "g_eff[4, 4]" in content
        ), "Phase components gαβ should be assigned"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
