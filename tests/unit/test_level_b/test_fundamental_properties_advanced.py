"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Advanced Level B fundamental properties tests for the 7D phase field theory.

This module implements advanced tests for fundamental properties of the
phase field in homogeneous "interval-free" medium, including topological
charge quantization and zone separation.

Theoretical Background:
    Tests validate advanced aspects of the phase field behavior governed by
    the Riesz operator L_β = μ(-Δ)^β + λ in homogeneous medium, including
    topological charge quantization and zone separation.

Example:
    >>> test_suite = LevelBFundamentalPropertiesTestsAdvanced()
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


def test_topological_charge():
    """Test B3: Topological charge quantization."""
    test_suite = LevelBFundamentalPropertiesTestsAdvanced()
    result = test_suite.test_topological_charge()
    assert result[
        "passed"
    ], f"Topological charge test failed: {result.get('error', 'Unknown error')}"


def test_zone_separation():
    """Test B4: Zone separation."""
    test_suite = LevelBFundamentalPropertiesTestsAdvanced()
    result = test_suite.test_zone_separation()
    assert result[
        "passed"
    ], f"Zone separation test failed: {result.get('error', 'Unknown error')}"


class LevelBFundamentalPropertiesTestsAdvanced:
    """
    Advanced tests for Level B fundamental properties.

    Physical Meaning:
        Tests verify advanced aspects of the phase field behavior
        in homogeneous medium, including topological charge quantization
        and zone separation.
    """

    def __init__(self):
        """Initialize test suite."""
        self.domain = Domain(
            L=80.0,
            N=256,
            N_phi=32,
            N_t=64,
            T=40.0,
            dimensions=7,
        )
        
        self.parameters = Parameters(
            mu=1.0,
            beta=1.5,
            lambda_param=0.1,
            nu=1.0,
        )
        
        # Convert Parameters to dict for BVPSource
        config = {
            "carrier_frequency": 1.85e43,
            "envelope_amplitude": 1.0,
            "base_source_type": "gaussian"
        }
        self.source = BVPSource(self.domain, config)
        self.power_law_analyzer = LevelBPowerLawAnalyzer()
        self.node_analyzer = LevelBNodeAnalyzer()
        self.zone_analyzer = LevelBZoneAnalyzer()

    def test_topological_charge(self) -> Dict[str, Any]:
        """
        Test B3: Topological charge quantization.
        
        Physical Meaning:
            Verifies that the phase field exhibits quantized
            topological charge in homogeneous medium.
            
        Mathematical Foundation:
            The topological charge should be quantized to
            integer values in the 7D phase field theory.
        """
        try:
            # Generate test field
            field = self._generate_test_field()
            
            # Analyze topological charge
            center = [self.domain.N//2, self.domain.N//2, self.domain.N//2]
            charge_result = self.node_analyzer.compute_topological_charge(field, center)
            
            # Verify charge quantization
            assert abs(charge_result["charge"] - round(charge_result["charge"])) < 1e-10, "Charge should be quantized"
            assert charge_result["charge"] != 0, "Charge should be non-zero"
            assert charge_result["passed"], "Charge analysis should pass"
            
            return {
                "passed": True,
                "charge": charge_result["charge"],
                "integer_charge": charge_result["integer_charge"],
                "error": charge_result["error"]
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": str(e)
            }

    def test_zone_separation(self) -> Dict[str, Any]:
        """
        Test B4: Zone separation.
        
        Physical Meaning:
            Verifies that the phase field exhibits
            clear zone separation in homogeneous medium.
            
        Mathematical Foundation:
            The phase field should exhibit distinct
            zones with different properties.
        """
        try:
            # Generate test field
            field = self._generate_test_field()
            
            # Analyze zone separation
            center = [self.domain.N//2, self.domain.N//2, self.domain.N//2]
            thresholds = {
                "N_core": 0.7,
                "S_core": 0.6,
                "N_tail": 0.2,
                "S_tail": 0.2,
            }
            zone_result = self.zone_analyzer.separate_zones(field, center, thresholds)
            
            # Verify zone separation
            assert zone_result["passed"], "Zone separation should pass"
            assert zone_result["r_core"] > 0, "Should have core radius"
            assert zone_result["r_tail"] > 0, "Should have tail radius"
            assert zone_result["r_core"] < zone_result["r_tail"], "Core should be inside tail"
            
            return {
                "passed": True,
                "r_core": zone_result["r_core"],
                "r_tail": zone_result["r_tail"],
                "r_transition": zone_result["r_transition"],
                "quality_metrics": zone_result["quality_metrics"]
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": str(e)
            }

    def test_advanced_power_law_analysis(self) -> Dict[str, Any]:
        """
        Test advanced power law analysis.
        
        Physical Meaning:
            Verifies advanced aspects of power law behavior
            including scaling and universality.
        """
        try:
            # Generate test field
            field = self._generate_test_field()
            
            # Analyze advanced power law properties
            advanced_result = self.power_law_analyzer.analyze_advanced_power_law(field)
            
            # Verify advanced properties
            assert advanced_result["scaling_exponent"] > 0, "Scaling exponent should be positive"
            assert advanced_result["universality"] > 0.8, "Universality should be high"
            assert advanced_result["critical_exponent"] > 0, "Critical exponent should be positive"
            
            return {
                "passed": True,
                "scaling_exponent": advanced_result["scaling_exponent"],
                "universality": advanced_result["universality"],
                "critical_exponent": advanced_result["critical_exponent"]
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": str(e)
            }

    def test_advanced_node_analysis(self) -> Dict[str, Any]:
        """
        Test advanced node analysis.
        
        Physical Meaning:
            Verifies advanced aspects of node behavior
            including topology and stability.
        """
        try:
            # Generate test field
            field = self._generate_test_field()
            
            # Analyze advanced node properties
            advanced_result = self.node_analyzer.analyze_advanced_nodes(field)
            
            # Verify advanced properties
            assert advanced_result["node_topology"] is not None, "Node topology should be defined"
            assert advanced_result["node_stability"] > 0.5, "Node stability should be reasonable"
            assert advanced_result["node_interactions"] >= 0, "Node interactions should be non-negative"
            
            return {
                "passed": True,
                "node_topology": advanced_result["node_topology"],
                "node_stability": advanced_result["node_stability"],
                "node_interactions": advanced_result["node_interactions"]
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": str(e)
            }

    def test_advanced_zone_analysis(self) -> Dict[str, Any]:
        """
        Test advanced zone analysis.
        
        Physical Meaning:
            Verifies advanced aspects of zone behavior
            including dynamics and transitions.
        """
        try:
            # Generate test field
            field = self._generate_test_field()
            
            # Analyze advanced zone properties
            center = [self.domain.N//2, self.domain.N//2, self.domain.N//2]
            thresholds = {
                "N_core": 0.7,
                "S_core": 0.6,
                "N_tail": 0.2,
                "S_tail": 0.2,
            }
            advanced_result = self.zone_analyzer.separate_zones(field, center, thresholds)
            
            # Verify advanced properties
            assert advanced_result["passed"], "Advanced zone analysis should pass"
            assert advanced_result["r_core"] > 0, "Should have core radius"
            assert advanced_result["r_tail"] > 0, "Should have tail radius"
            assert advanced_result["r_core"] < advanced_result["r_tail"], "Core should be inside tail"
            
            return {
                "passed": True,
                "r_core": advanced_result["r_core"],
                "r_tail": advanced_result["r_tail"],
                "r_transition": advanced_result["r_transition"],
                "quality_metrics": advanced_result["quality_metrics"]
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": str(e)
            }

    def _generate_test_field(self) -> np.ndarray:
        """Generate test field for analysis."""
        # Create source field
        source_field = self.source.generate()
        
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
        """Run all advanced tests."""
        results = {}
        
        # Test B3: Topological charge
        results["topological_charge"] = self.test_topological_charge()
        
        # Test B4: Zone separation
        results["zone_separation"] = self.test_zone_separation()
        
        # Advanced tests
        results["advanced_power_law"] = self.test_advanced_power_law_analysis()
        results["advanced_nodes"] = self.test_advanced_node_analysis()
        results["advanced_zones"] = self.test_advanced_zone_analysis()
        
        return results


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])
