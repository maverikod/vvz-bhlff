"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Basic Level B fundamental properties tests for the 7D phase field theory.

This module implements basic tests for fundamental properties of the
phase field in homogeneous "interval-free" medium, validating the core
theoretical predictions of the 7D phase field theory.

Theoretical Background:
    Tests validate the fundamental behavior of the phase field governed by
    the Riesz operator L_β = μ(-Δ)^β + λ in homogeneous medium, including
    power law tails and absence of spherical nodes.

Example:
    >>> test_suite = LevelBFundamentalPropertiesTestsBasic()
    >>> results = test_suite.run_all_tests()
"""

import numpy as np
import pytest
import unittest
from typing import Dict, Any, Tuple, List
from scipy import stats
import json
import matplotlib.pyplot as plt
from pathlib import Path

from bhlff.core.domain import Domain
from bhlff.core.domain.parameters import Parameters
from bhlff.core.sources.bvp_source import BVPSource
from bhlff.models.level_b.power_law_analyzer import LevelBPowerLawAnalyzer
from bhlff.models.level_b.node_analyzer import LevelBNodeAnalyzer
from bhlff.models.level_b.zone_analyzer import LevelBZoneAnalyzer


def test_power_law_tail():
    """Test B1: Power law tail in homogeneous medium."""
    test_suite = LevelBFundamentalPropertiesTestsBasic()
    result = test_suite.test_power_law_tail()
    assert result[
        "passed"
    ], f"Power law test failed: {result.get('error', 'Unknown error')}"


def test_no_spherical_nodes():
    """Test B2: Absence of spherical standing nodes."""
    test_suite = LevelBFundamentalPropertiesTestsBasic()
    result = test_suite.test_no_spherical_nodes()
    assert result[
        "passed"
    ], f"No spherical nodes test failed: {result.get('error', 'Unknown error')}"


class LevelBFundamentalPropertiesTestsBasic:
    """
    Basic tests for Level B fundamental properties.

    Physical Meaning:
        Tests verify the basic fundamental behavior of the phase field
        in homogeneous medium, including power law tails and node absence.
    """

    def __init__(self):
        """Initialize test suite."""
        self.domain = Domain(
            L=20.0,
            N=64,
            N_phi=8,
            N_t=16,
            T=10.0,
            dimensions=7,
        )
        
        self.parameters = Parameters(
            mu=1.0,
            beta=1.5,
            lambda_param=0.1,
            nu=1.0,
        )
        
        self.source = BVPSource(self.domain, self.parameters)
        self.power_law_analyzer = LevelBPowerLawAnalyzer(self.domain, self.parameters)
        self.node_analyzer = LevelBNodeAnalyzer(self.domain, self.parameters)
        self.zone_analyzer = LevelBZoneAnalyzer(self.domain, self.parameters)

    def test_power_law_tail(self):
        """
        Test B1: Power law tail in homogeneous medium.
        
        Physical Meaning:
            Verifies that the phase field exhibits power law behavior
            in the tail region of homogeneous medium.
            
        Mathematical Foundation:
            The Riesz operator L_β = μ(-Δ)^β + λ should produce
            power law tails with exponent related to β.
        """
        try:
            # Generate test field
            field = self._generate_test_field()
            
            # Analyze power law behavior
            power_law_result = self.power_law_analyzer.analyze_power_law_tail(field)
            
            # Verify power law properties
            assert power_law_result["exponent"] > 0, "Power law exponent should be positive"
            assert power_law_result["r_squared"] > 0.8, "Power law fit should be good"
            assert power_law_result["tail_length"] > 0, "Tail length should be positive"
            
            return {
                "passed": True,
                "exponent": power_law_result["exponent"],
                "r_squared": power_law_result["r_squared"],
                "tail_length": power_law_result["tail_length"]
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": str(e)
            }

    def test_no_spherical_nodes(self) -> Dict[str, Any]:
        """
        Test B2: Absence of spherical standing nodes.
        
        Physical Meaning:
            Verifies that the phase field does not exhibit
            spherical standing nodes in homogeneous medium.
            
        Mathematical Foundation:
            The Riesz operator should not produce spherical
            nodes in the homogeneous case.
        """
        try:
            # Generate test field
            field = self._generate_test_field()
            
            # Analyze nodes
            node_result = self.node_analyzer.analyze_spherical_nodes(field)
            
            # Verify no spherical nodes
            assert node_result["spherical_nodes_count"] == 0, "Should have no spherical nodes"
            assert node_result["node_density"] < 0.1, "Node density should be low"
            assert node_result["spherical_symmetry"] < 0.5, "Spherical symmetry should be low"
            
            return {
                "passed": True,
                "spherical_nodes_count": node_result["spherical_nodes_count"],
                "node_density": node_result["node_density"],
                "spherical_symmetry": node_result["spherical_symmetry"]
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": str(e)
            }

    def _generate_test_field(self) -> np.ndarray:
        """Generate test field for analysis."""
        # Create source field
        source_field = self.source.generate_source_field()
        
        # Solve the field equation
        field = self._solve_field_equation(source_field)
        
        return field

    def _solve_field_equation(self, source: np.ndarray) -> np.ndarray:
        """Solve the field equation L_β a = s."""
        # This is a simplified solver for testing
        # In practice, this would use the full FFT solver
        
        # Create spectral coefficients
        kx = np.fft.fftfreq(self.domain.N, self.domain.L / self.domain.N)
        ky = np.fft.fftfreq(self.domain.N, self.domain.L / self.domain.N)
        kz = np.fft.fftfreq(self.domain.N, self.domain.L / self.domain.N)
        
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        k_magnitude = np.sqrt(KX**2 + KY**2 + KZ**2)
        
        # Compute spectral coefficients
        mu = self.parameters.mu
        beta = self.parameters.beta
        lambda_param = self.parameters.lambda_param
        
        spectral_coeffs = mu * (k_magnitude ** (2 * beta)) + lambda_param
        
        # Handle k=0 mode
        if lambda_param == 0:
            spectral_coeffs[0, 0, 0] = 1.0
        
        # Transform to spectral space
        source_spectral = np.fft.fftn(source)
        
        # Apply spectral operator
        solution_spectral = source_spectral / spectral_coeffs
        
        # Transform back to real space
        solution = np.fft.ifftn(solution_spectral)
        
        return solution.real

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all basic tests."""
        results = {}
        
        # Test B1: Power law tail
        results["power_law_tail"] = self.test_power_law_tail()
        
        # Test B2: No spherical nodes
        results["no_spherical_nodes"] = self.test_no_spherical_nodes()
        
        return results


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])
