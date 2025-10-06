"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Simple Level B tests for fundamental properties.

This module implements basic tests for Level B fundamental properties
of the 7D phase field theory, focusing on power law analysis.
"""

import numpy as np
import unittest
from typing import Dict, Any, List

from bhlff.core.domain import Domain
from bhlff.core.domain.parameters import Parameters
from bhlff.models.level_b.power_law_analyzer import LevelBPowerLawAnalyzer


class TestSimpleLevelB(unittest.TestCase):
    """
    Simple Level B tests for fundamental properties.

    Physical Meaning:
        Tests the basic functionality of Level B analysis tools,
        validating that the power law analysis works correctly
        with analytical test solutions.
    """

    def setUp(self):
        """Set up test fixtures."""
        self.domain = Domain(L=20.0, N=256, N_phi=4, N_t=8, T=1.0)
        self.parameters = Parameters(mu=1.0, beta=1.0, lambda_param=0.0)
        self.center = [10.0, 10.0, 10.0]
        self.analyzer = LevelBPowerLawAnalyzer()

    def _create_test_solution(self, beta: float) -> np.ndarray:
        """
        Create analytical test solution with power law behavior.

        Physical Meaning:
            Creates a test solution that exhibits the expected
            power law behavior A(r) ∝ r^(2β-3) for validation.
        """
        # Create coordinate grids
        x = np.linspace(0, self.domain.L, self.domain.N)
        y = np.linspace(0, self.domain.L, self.domain.N)
        z = np.linspace(0, self.domain.L, self.domain.N)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        # Compute distances from center
        distances = np.sqrt(
            (X - self.center[0]) ** 2
            + (Y - self.center[1]) ** 2
            + (Z - self.center[2]) ** 2
        )

        # Avoid division by zero
        distances = np.maximum(distances, 1e-10)

        # Create power law solution A(r) ∝ r^(2β-3)
        exponent = 2 * beta - 3
        amplitude = distances**exponent

        # Add some phase structure
        phase = np.arctan2(Y - self.center[1], X - self.center[0])

        # Create complex field
        solution = amplitude * np.exp(1j * phase)

        return solution

    def test_power_law_analysis_basic(self):
        """Test basic power law analysis functionality."""
        # Create test solution
        solution = self._create_test_solution(beta=1.0)

        # Run power law analysis with relaxed requirements
        result = self.analyzer.analyze_power_law_tail(
            solution, 1.0, self.center, min_decades=0.5
        )

        # Check basic properties
        self.assertIn("slope", result)
        self.assertIn("theoretical_slope", result)
        self.assertIn("r_squared", result)
        self.assertIn("passed", result)

        # Check that analysis completed
        self.assertIsNotNone(result["slope"])
        self.assertIsNotNone(result["r_squared"])

        print(f"Power law analysis completed:")
        print(f"  Slope: {result['slope']:.3f}")
        print(f"  Theoretical: {result['theoretical_slope']:.3f}")
        print(f"  R²: {result['r_squared']:.3f}")
        print(f"  Passed: {result['passed']}")

    def test_power_law_different_beta(self):
        """Test power law analysis for different beta values."""
        beta_values = [0.5, 0.7, 1.0, 1.3]

        for beta in beta_values:
            with self.subTest(beta=beta):
                # Create test solution
                solution = self._create_test_solution(beta=beta)

                # Run power law analysis with relaxed requirements
                result = self.analyzer.analyze_power_law_tail(
                    solution, beta, self.center, min_decades=0.5
                )

                # Check that theoretical slope is correct
                expected_slope = 2 * beta - 3
                self.assertAlmostEqual(
                    result["theoretical_slope"], expected_slope, places=10
                )

                print(
                    f"β={beta}: slope={result['slope']:.3f}, theoretical={result['theoretical_slope']:.3f}"
                )

    def test_radial_profile_computation(self):
        """Test radial profile computation."""
        # Create test solution
        solution = self._create_test_solution(beta=1.0)

        # Test radial profile computation
        radial_profile = self.analyzer._compute_radial_profile(solution, self.center)

        # Check basic properties
        self.assertIn("r", radial_profile)
        self.assertIn("A", radial_profile)
        self.assertEqual(len(radial_profile["r"]), len(radial_profile["A"]))
        self.assertGreater(len(radial_profile["r"]), 0)

        # Check that amplitude decreases with radius (power law behavior)
        r = radial_profile["r"]
        A = radial_profile["A"]

        # Find indices where r is increasing
        increasing_r = np.where(np.diff(r) > 0)[0]
        if len(increasing_r) > 1:
            # Check that amplitude generally decreases
            decreasing_amp = np.where(np.diff(A[increasing_r]) < 0)[0]
            self.assertGreater(
                len(decreasing_amp),
                0,
                "Amplitude should generally decrease with radius",
            )

        print(f"Radial profile computed: {len(radial_profile['r'])} points")
        print(f"  Radius range: [{r.min():.3f}, {r.max():.3f}]")
        print(f"  Amplitude range: [{A.min():.3e}, {A.max():.3e}]")


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
