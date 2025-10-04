"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Level C integration module for comprehensive boundary and cell analysis.

This module provides integrated analysis capabilities for Level C tests,
including boundary effects, resonator chains, quench memory, and mode
beating analysis in the 7D phase field theory.

Physical Meaning:
    Integrates all Level C analysis capabilities:
    - C1: Single wall boundary effects and resonance mode analysis
    - C2: Resonator chain analysis with ABCD model validation
    - C3: Quench memory and pinning effects analysis
    - C4: Mode beating and drift velocity analysis

Mathematical Foundation:
    Implements comprehensive Level C analysis:
    - Boundary analysis: Y(ω) = I(ω)/V(ω), A(r) = (1/4π) ∫_S(r) |a(x)|² dS
    - ABCD model: T_total = ∏ T_ℓ, det(T_total - I) = 0
    - Memory analysis: Γ_memory[a] = -γ ∫_0^t K(t-τ) a(τ) dτ
    - Beating analysis: v_cell^pred = Δω / |k₂ - k₁|

Example:
    >>> integrator = LevelCIntegration(bvp_core)
    >>> results = integrator.run_all_tests(domain, test_params)
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
from dataclasses import dataclass

from bhlff.core.bvp import BVPCore
from .boundary_analysis import BoundaryAnalysis
from .abcd_model import ABCDModel, ResonatorLayer
from .quench_memory_analysis import QuenchMemoryAnalysis
from .mode_beating_analysis import ModeBeatingAnalysis


@dataclass
class LevelCResults:
    """
    Level C test results.

    Physical Meaning:
        Contains the results of all Level C tests,
        including individual test results and overall
        validation status.
    """

    c1_results: Dict[str, Any]
    c2_results: Dict[str, Any]
    c3_results: Dict[str, Any]
    c4_results: Dict[str, Any]
    overall_status: str
    validation_passed: bool


@dataclass
class TestConfiguration:
    """
    Test configuration for Level C tests.

    Physical Meaning:
        Defines the configuration parameters for all
        Level C tests, including domain, physics,
        and analysis parameters.
    """

    domain: Dict[str, Any]
    physics: Dict[str, Any]
    c1_params: Dict[str, Any]
    c2_params: Dict[str, Any]
    c3_params: Dict[str, Any]
    c4_params: Dict[str, Any]


class LevelCIntegration:
    """
    Level C integration for comprehensive boundary and cell analysis.

    Physical Meaning:
        Integrates all Level C analysis capabilities for
        comprehensive boundary and cell analysis in the
        7D phase field theory.

    Mathematical Foundation:
        Implements comprehensive Level C analysis:
        - C1: Boundary effects and resonance mode analysis
        - C2: Resonator chain analysis with ABCD validation
        - C3: Quench memory and pinning effects analysis
        - C4: Mode beating and drift velocity analysis
    """

    def __init__(self, bvp_core: BVPCore):
        """
        Initialize Level C integration.

        Args:
            bvp_core (BVPCore): BVP core framework instance.
        """
        self.bvp_core = bvp_core
        self.logger = logging.getLogger(__name__)

        # Initialize analysis modules
        self.boundary_analyzer = BoundaryAnalysis(bvp_core)
        self.memory_analyzer = QuenchMemoryAnalysis(bvp_core)
        self.beating_analyzer = ModeBeatingAnalysis(bvp_core)

    def run_all_tests(self, test_config: TestConfiguration) -> LevelCResults:
        """
        Run all Level C tests.

        Physical Meaning:
            Executes all Level C tests in sequence, providing
            comprehensive analysis of boundary effects, resonator
            chains, memory effects, and mode beating.

        Mathematical Foundation:
            Runs all four Level C tests:
            - C1: Single wall boundary analysis
            - C2: Resonator chain ABCD analysis
            - C3: Quench memory and pinning analysis
            - C4: Mode beating and drift analysis

        Args:
            test_config (TestConfiguration): Test configuration.

        Returns:
            LevelCResults: Comprehensive Level C test results.
        """
        self.logger.info("Starting comprehensive Level C analysis")

        # Run C1: Single wall boundary analysis
        self.logger.info("Running C1: Single wall boundary analysis")
        c1_results = self._run_c1_test(test_config)

        # Run C2: Resonator chain ABCD analysis
        self.logger.info("Running C2: Resonator chain ABCD analysis")
        c2_results = self._run_c2_test(test_config)

        # Run C3: Quench memory and pinning analysis
        self.logger.info("Running C3: Quench memory and pinning analysis")
        c3_results = self._run_c3_test(test_config)

        # Run C4: Mode beating analysis
        self.logger.info("Running C4: Mode beating analysis")
        c4_results = self._run_c4_test(test_config)

        # Validate overall results
        overall_status = self._validate_overall_results(
            c1_results, c2_results, c3_results, c4_results
        )
        validation_passed = overall_status == "PASS"

        # Create comprehensive results
        results = LevelCResults(
            c1_results=c1_results,
            c2_results=c2_results,
            c3_results=c3_results,
            c4_results=c4_results,
            overall_status=overall_status,
            validation_passed=validation_passed,
        )

        self.logger.info(f"Level C analysis completed with status: {overall_status}")
        return results

    def _run_c1_test(self, test_config: TestConfiguration) -> Dict[str, Any]:
        """
        Run C1: Single wall boundary analysis.

        Physical Meaning:
            Performs single wall boundary analysis to study
            resonance mode birth and admittance contrast effects.
        """
        try:
            # Extract C1 parameters
            c1_params = test_config.c1_params

            # Run boundary analysis
            c1_results = self.boundary_analyzer.analyze_single_wall(
                test_config.domain, c1_params
            )

            # Validate C1 results
            c1_valid = self._validate_c1_results(c1_results)
            c1_results["validation_passed"] = c1_valid

            return c1_results

        except Exception as e:
            self.logger.error(f"C1 test failed: {e}")
            return {
                "error": str(e),
                "validation_passed": False,
                "test_status": "FAILED",
            }

    def _run_c2_test(self, test_config: TestConfiguration) -> Dict[str, Any]:
        """
        Run C2: Resonator chain ABCD analysis.

        Physical Meaning:
            Performs resonator chain analysis using ABCD model
            to study system resonance modes and coupling effects.
        """
        try:
            # Extract C2 parameters
            c2_params = test_config.c2_params

            # Create resonator layers
            resonator_layers = self._create_resonator_layers(c2_params)

            # Create ABCD model
            abcd_model = ABCDModel(resonator_layers, self.bvp_core)

            # Run ABCD analysis
            c2_results = self._run_abcd_analysis(abcd_model, c2_params)

            # Validate C2 results
            c2_valid = self._validate_c2_results(c2_results)
            c2_results["validation_passed"] = c2_valid

            return c2_results

        except Exception as e:
            self.logger.error(f"C2 test failed: {e}")
            return {
                "error": str(e),
                "validation_passed": False,
                "test_status": "FAILED",
            }

    def _run_c3_test(self, test_config: TestConfiguration) -> Dict[str, Any]:
        """
        Run C3: Quench memory and pinning analysis.

        Physical Meaning:
            Performs quench memory analysis to study memory
            effects, pinning, and field stabilization.
        """
        try:
            # Extract C3 parameters
            c3_params = test_config.c3_params

            # Run memory analysis
            c3_results = self.memory_analyzer.analyze_quench_memory(
                test_config.domain, c3_params
            )

            # Validate C3 results
            c3_valid = self._validate_c3_results(c3_results)
            c3_results["validation_passed"] = c3_valid

            return c3_results

        except Exception as e:
            self.logger.error(f"C3 test failed: {e}")
            return {
                "error": str(e),
                "validation_passed": False,
                "test_status": "FAILED",
            }

    def _run_c4_test(self, test_config: TestConfiguration) -> Dict[str, Any]:
        """
        Run C4: Mode beating analysis.

        Physical Meaning:
            Performs mode beating analysis to study dual-mode
            excitation, beating patterns, and drift velocity.
        """
        try:
            # Extract C4 parameters
            c4_params = test_config.c4_params

            # Run beating analysis
            c4_results = self.beating_analyzer.analyze_mode_beating(
                test_config.domain, c4_params
            )

            # Validate C4 results
            c4_valid = self._validate_c4_results(c4_results)
            c4_results["validation_passed"] = c4_valid

            return c4_results

        except Exception as e:
            self.logger.error(f"C4 test failed: {e}")
            return {
                "error": str(e),
                "validation_passed": False,
                "test_status": "FAILED",
            }

    def _create_resonator_layers(
        self, c2_params: Dict[str, Any]
    ) -> List[ResonatorLayer]:
        """
        Create resonator layers for C2 test.

        Physical Meaning:
            Creates resonator layers based on C2 parameters
            for ABCD model analysis.
        """
        shells = c2_params.get("shells", [])
        resonator_layers = []

        for shell in shells:
            layer = ResonatorLayer(
                radius=shell.get("radius", 1.0),
                thickness=shell.get("thickness", 0.1),
                contrast=shell.get("contrast", 0.5),
                memory_gamma=shell.get("memory_gamma", 0.0),
                memory_tau=shell.get("memory_tau", 1.0),
            )
            resonator_layers.append(layer)

        return resonator_layers

    def _run_abcd_analysis(
        self, abcd_model: ABCDModel, c2_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run ABCD analysis for C2 test.

        Physical Meaning:
            Performs ABCD model analysis to study system
            resonance modes and coupling effects.
        """
        # Extract frequency range
        frequency_range = c2_params.get("frequency_range", (0.05, 6.0))

        # Find system modes
        system_modes = abcd_model.find_system_modes(frequency_range)

        # Compute system admittance
        frequencies = np.logspace(
            np.log10(frequency_range[0]), np.log10(frequency_range[1]), 100
        )
        admittance = [
            abcd_model.compute_system_admittance(freq) for freq in frequencies
        ]

        # Create numerical results for comparison
        numerical_results = {
            "frequencies": frequencies,
            "admittance": admittance,
            "modes": [
                {"frequency": mode.frequency, "quality_factor": mode.quality_factor}
                for mode in system_modes
            ],
        }

        # Compare with numerical results
        comparison = abcd_model.compare_with_numerical(numerical_results)

        return {
            "system_modes": system_modes,
            "admittance_spectrum": {
                "frequencies": frequencies,
                "admittance": admittance,
            },
            "abcd_comparison": comparison,
            "test_passed": comparison.get("comparison_passed", False),
        }

    def _validate_c1_results(self, c1_results: Dict[str, Any]) -> bool:
        """
        Validate C1 test results.

        Physical Meaning:
            Validates that C1 test results meet the acceptance
            criteria for boundary analysis.
        """
        if "error" in c1_results:
            return False

        # Check resonance birth threshold
        resonance_threshold = c1_results.get("resonance_threshold", float("inf"))
        if resonance_threshold > 0.1:
            return False

        # Check that resonances appear at higher contrasts
        contrast_results = c1_results.get("contrast_results", {})
        high_contrast_results = [
            result
            for key, result in contrast_results.items()
            if result.get("contrast", 0) >= 0.1
        ]
        if not any(
            result.get("has_resonances", False) for result in high_contrast_results
        ):
            return False

        return True

    def _validate_c2_results(self, c2_results: Dict[str, Any]) -> bool:
        """
        Validate C2 test results.

        Physical Meaning:
            Validates that C2 test results meet the acceptance
            criteria for resonator chain analysis.
        """
        if "error" in c2_results:
            return False

        # Check ABCD comparison
        abcd_comparison = c2_results.get("abcd_comparison", {})
        if not abcd_comparison.get("comparison_passed", False):
            return False

        # Check error thresholds
        max_frequency_error = abcd_comparison.get("max_frequency_error", 1.0)
        max_quality_error = abcd_comparison.get("max_quality_error", 1.0)

        if max_frequency_error > 0.05 or max_quality_error > 0.10:
            return False

        return True

    def _validate_c3_results(self, c3_results: Dict[str, Any]) -> bool:
        """
        Validate C3 test results.

        Physical Meaning:
            Validates that C3 test results meet the acceptance
            criteria for quench memory analysis.
        """
        if "error" in c3_results:
            return False

        # Check freezing threshold
        freezing_threshold = c3_results.get("freezing_threshold", float("inf"))
        if freezing_threshold > 0.1:
            return False

        # Check that pinning occurs at higher memory strengths
        memory_results = c3_results.get("memory_results", {})
        high_memory_results = [
            result
            for key, result in memory_results.items()
            if result.get("gamma", 0) >= 0.4
        ]
        if not any(result.get("is_pinned", False) for result in high_memory_results):
            return False

        return True

    def _validate_c4_results(self, c4_results: Dict[str, Any]) -> bool:
        """
        Validate C4 test results.

        Physical Meaning:
            Validates that C4 test results meet the acceptance
            criteria for mode beating analysis.
        """
        if "error" in c4_results:
            return False

        # Check beating results
        beating_results = c4_results.get("beating_results", {})
        for key, result in beating_results.items():
            error_analysis = result.get("error_analysis", {})

            # Check background error (should be ≤ 10%)
            background_error = error_analysis.get("background_error", 1.0)
            if background_error > 0.10:
                return False

            # Check suppression factor (should be ≥ 10×)
            suppression_factor = error_analysis.get("suppression_factor", 1.0)
            if suppression_factor > 0.1:
                return False

        return True

    def _validate_overall_results(
        self,
        c1_results: Dict[str, Any],
        c2_results: Dict[str, Any],
        c3_results: Dict[str, Any],
        c4_results: Dict[str, Any],
    ) -> str:
        """
        Validate overall Level C results.

        Physical Meaning:
            Validates that all Level C tests meet their
            acceptance criteria and determines overall status.
        """
        # Check individual test results
        c1_passed = c1_results.get("validation_passed", False)
        c2_passed = c2_results.get("validation_passed", False)
        c3_passed = c3_results.get("validation_passed", False)
        c4_passed = c4_results.get("validation_passed", False)

        # Determine overall status
        if c1_passed and c2_passed and c3_passed and c4_passed:
            return "PASS"
        else:
            return "FAIL"

    def create_test_configuration(
        self, domain_params: Dict[str, Any], physics_params: Dict[str, Any]
    ) -> TestConfiguration:
        """
        Create test configuration for Level C tests.

        Physical Meaning:
            Creates a comprehensive test configuration for
            all Level C tests based on domain and physics parameters.
        """
        # C1 parameters
        c1_params = {
            "contrast_range": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            "frequency_range": (0.05, 5.0),
            "boundary": {
                "center": [
                    domain_params["L"] / 2,
                    domain_params["L"] / 2,
                    domain_params["L"] / 2,
                ],
                "radius": domain_params["L"] / 6,
                "thickness": 3,
            },
        }

        # C2 parameters
        c2_params = {
            "frequency_range": (0.05, 6.0),
            "shells": [
                {
                    "radius": 1.0,
                    "thickness": 0.1,
                    "contrast": 0.3,
                    "memory_gamma": 0.0,
                    "memory_tau": 1.0,
                },
                {
                    "radius": 2.0,
                    "thickness": 0.1,
                    "contrast": 0.5,
                    "memory_gamma": 0.0,
                    "memory_tau": 1.0,
                },
                {
                    "radius": 3.0,
                    "thickness": 0.1,
                    "contrast": 0.7,
                    "memory_gamma": 0.0,
                    "memory_tau": 1.0,
                },
            ],
        }

        # C3 parameters
        c3_params = {
            "gamma_list": [0.0, 0.2, 0.4, 0.6, 0.8],
            "tau_list": [0.5, 1.0, 2.0],
            "time_integration": {"dt": 0.005, "T": 400.0, "avg_window": 0.8},
        }

        # C4 parameters
        c4_params = {
            "omega_0": 1.0,
            "delta_omega_ratios": [0.02, 0.05],
            "time_integration": {"dt": 0.005, "T": 400.0, "avg_window": 0.8},
        }

        return TestConfiguration(
            domain=domain_params,
            physics=physics_params,
            c1_params=c1_params,
            c2_params=c2_params,
            c3_params=c3_params,
            c4_params=c4_params,
        )
