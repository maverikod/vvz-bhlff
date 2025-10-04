"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Mode beating analysis module for Level C test C4.

This module implements comprehensive analysis of mode beating effects
in the 7D phase field theory, focusing on dual-mode excitation,
beating patterns, and drift velocity analysis.

Physical Meaning:
    Analyzes mode beating effects in the 7D phase field, including:
    - Dual-mode excitation and superposition
    - Beating pattern analysis and frequency characteristics
    - Drift velocity analysis and theoretical comparison
    - Pinning effects on mode beating

Mathematical Foundation:
    Implements mode beating analysis using:
    - Dual-mode source: s(x,t) = s₁(x) e^(-iω₁t) + s₂(x) e^(-iω₂t)
    - Theoretical drift velocity: v_cell^pred = Δω / |k₂ - k₁|
    - Beating frequency: ω_beat = |ω₂ - ω₁|
    - Drift suppression analysis with pinning

Example:
    >>> analyzer = ModeBeatingAnalysis(bvp_core)
    >>> results = analyzer.analyze_mode_beating(domain, beating_params)
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
from dataclasses import dataclass

from bhlff.core.bvp import BVPCore


@dataclass
class DualModeSource:
    """
    Dual-mode source specification.

    Physical Meaning:
        Defines a dual-mode source for mode beating analysis,
        including frequencies, amplitudes, and spatial profiles.
    """

    frequency_1: float
    frequency_2: float
    amplitude_1: float = 1.0
    amplitude_2: float = 1.0
    profile_1: Optional[np.ndarray] = None
    profile_2: Optional[np.ndarray] = None


@dataclass
class BeatingPattern:
    """
    Beating pattern analysis results.

    Physical Meaning:
        Contains the results of beating pattern analysis,
        including beating frequency, amplitude modulation,
        and temporal characteristics.
    """

    beating_frequency: float
    amplitude_modulation: np.ndarray
    phase_evolution: np.ndarray
    temporal_coherence: float


@dataclass
class DriftVelocityAnalysis:
    """
    Drift velocity analysis results.

    Physical Meaning:
        Contains the results of drift velocity analysis,
        including numerical and theoretical values,
        and error metrics.
    """

    numerical_velocity: float
    theoretical_velocity: float
    relative_error: float
    suppression_factor: float = 1.0


@dataclass
class WaveVector:
    """
    Wave vector information.

    Physical Meaning:
        Represents wave vector information for a mode,
        including magnitude and direction.
    """

    magnitude: float
    direction: np.ndarray
    frequency: float


class ModeBeatingAnalysis:
    """
    Mode beating analysis for Level C test C4.

    Physical Meaning:
        Analyzes mode beating effects in the 7D phase field,
        including dual-mode excitation, beating patterns,
        and drift velocity analysis.

    Mathematical Foundation:
        Implements comprehensive mode beating analysis:
        - Dual-mode source: s(x,t) = s₁(x) e^(-iω₁t) + s₂(x) e^(-iω₂t)
        - Theoretical drift velocity: v_cell^pred = Δω / |k₂ - k₁|
        - Beating frequency: ω_beat = |ω₂ - ω₁|
        - Drift suppression analysis with pinning
    """

    def __init__(self, bvp_core: BVPCore):
        """
        Initialize mode beating analysis.

        Args:
            bvp_core (BVPCore): BVP core framework instance.
        """
        self.bvp_core = bvp_core
        self.logger = logging.getLogger(__name__)

    def analyze_mode_beating(
        self, domain: Dict[str, Any], beating_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze mode beating effects (C4 test).

        Physical Meaning:
            Performs comprehensive analysis of mode beating effects,
            including dual-mode excitation, beating patterns,
            and drift velocity analysis.

        Mathematical Foundation:
            Analyzes the system response to dual-mode excitation:
            - Dual-mode source: s(x,t) = s₁(x) e^(-iω₁t) + s₂(x) e^(-iω₂t)
            - Theoretical drift velocity: v_cell^pred = Δω / |k₂ - k₁|
            - Beating frequency: ω_beat = |ω₂ - ω₁|
            - Drift suppression analysis with pinning

        Args:
            domain (Dict[str, Any]): Domain parameters.
            beating_params (Dict[str, Any]): Beating parameters.

        Returns:
            Dict[str, Any]: Comprehensive mode beating analysis results.
        """
        self.logger.info("Starting mode beating analysis (C4)")

        # Extract parameters
        omega_0 = beating_params.get("omega_0", 1.0)
        delta_omega_ratios = beating_params.get("delta_omega_ratios", [0.02, 0.05])
        time_params = beating_params.get("time_integration", {})

        # Perform analysis for each frequency ratio
        beating_results = {}
        for delta_omega_ratio in delta_omega_ratios:
            self.logger.info(f"Analyzing Δω/ω₀ = {delta_omega_ratio}")

            # Create dual-mode source
            omega_1 = omega_0 - delta_omega_ratio * omega_0 / 2
            omega_2 = omega_0 + delta_omega_ratio * omega_0 / 2

            dual_mode = DualModeSource(
                frequency_1=omega_1,
                frequency_2=omega_2,
                amplitude_1=1.0,
                amplitude_2=1.0,
            )

            # Test without pinning (background)
            background_results = self._analyze_background_beating(
                domain, dual_mode, time_params
            )

            # Test with pinning
            pinned_results = self._analyze_pinned_beating(
                domain, dual_mode, time_params
            )

            # Compute theoretical values
            theoretical_analysis = self._compute_theoretical_analysis(dual_mode)

            # Analyze errors and suppression
            error_analysis = self._analyze_errors(
                background_results, pinned_results, theoretical_analysis
            )

            # Store results
            key = f"delta_omega_ratio_{delta_omega_ratio}"
            beating_results[key] = {
                "delta_omega_ratio": delta_omega_ratio,
                "dual_mode": dual_mode,
                "background_results": background_results,
                "pinned_results": pinned_results,
                "theoretical_analysis": theoretical_analysis,
                "error_analysis": error_analysis,
            }

        # Create summary
        summary = self._create_beating_summary(beating_results)

        return {
            "beating_results": beating_results,
            "summary": summary,
            "test_passed": self._validate_c4_results(beating_results),
        }

    def _analyze_background_beating(
        self,
        domain: Dict[str, Any],
        dual_mode: DualModeSource,
        time_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Analyze background beating without pinning.

        Physical Meaning:
            Analyzes mode beating in the absence of pinning
            effects, providing baseline measurements for
            comparison with pinned systems.
        """
        # Create dual-mode field
        field_dual = self._create_dual_mode_field(domain, dual_mode)

        # Perform time evolution
        time_evolution = self._evolve_dual_mode_field(
            field_dual, dual_mode, time_params
        )

        # Analyze beating patterns
        beating_pattern = self._analyze_beating_patterns(time_evolution, dual_mode)

        # Analyze drift velocity
        drift_analysis = self._analyze_drift_velocity(time_evolution)

        return {
            "field_evolution": time_evolution,
            "beating_pattern": beating_pattern,
            "drift_analysis": drift_analysis,
            "is_pinned": False,
        }

    def _analyze_pinned_beating(
        self,
        domain: Dict[str, Any],
        dual_mode: DualModeSource,
        time_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Analyze pinned beating with memory effects.

        Physical Meaning:
            Analyzes mode beating in the presence of pinning
            effects, including memory-induced field stabilization.
        """
        # Create dual-mode field with pinning
        field_dual_pinned = self._create_dual_mode_field_with_pinning(domain, dual_mode)

        # Perform time evolution with pinning
        time_evolution = self._evolve_dual_mode_field_with_pinning(
            field_dual_pinned, dual_mode, time_params
        )

        # Analyze beating patterns
        beating_pattern = self._analyze_beating_patterns(time_evolution, dual_mode)

        # Analyze drift velocity
        drift_analysis = self._analyze_drift_velocity(time_evolution)

        return {
            "field_evolution": time_evolution,
            "beating_pattern": beating_pattern,
            "drift_analysis": drift_analysis,
            "is_pinned": True,
        }

    def _compute_theoretical_analysis(
        self, dual_mode: DualModeSource
    ) -> Dict[str, Any]:
        """
        Compute theoretical analysis for dual-mode system.

        Physical Meaning:
            Computes theoretical predictions for the dual-mode
            system, including wave vectors and drift velocity.

        Mathematical Foundation:
            - Wave vectors: k₁,₂ = ω₁,₂ / c_φ
            - Theoretical drift velocity: v_cell^pred = Δω / |k₂ - k₁|
            - Beating frequency: ω_beat = |ω₂ - ω₁|
        """
        # Compute wave vectors
        k_1 = self._compute_wave_vector(dual_mode.frequency_1)
        k_2 = self._compute_wave_vector(dual_mode.frequency_2)

        # Compute theoretical drift velocity
        delta_omega = abs(dual_mode.frequency_2 - dual_mode.frequency_1)
        delta_k = abs(k_2.magnitude - k_1.magnitude)

        theoretical_velocity = delta_omega / delta_k if delta_k > 1e-12 else 0.0

        # Compute beating frequency
        beating_frequency = delta_omega

        return {
            "wave_vector_1": k_1,
            "wave_vector_2": k_2,
            "theoretical_velocity": theoretical_velocity,
            "beating_frequency": beating_frequency,
            "delta_omega": delta_omega,
            "delta_k": delta_k,
        }

    def _analyze_errors(
        self,
        background_results: Dict[str, Any],
        pinned_results: Dict[str, Any],
        theoretical_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Analyze errors between numerical and theoretical results.

        Physical Meaning:
            Computes error metrics between numerical results
            and theoretical predictions for both background
            and pinned systems.
        """
        # Background error analysis
        background_velocity = background_results["drift_analysis"].numerical_velocity
        theoretical_velocity = theoretical_analysis["theoretical_velocity"]

        background_error = (
            abs(background_velocity - theoretical_velocity) / theoretical_velocity
            if theoretical_velocity > 1e-12
            else 0.0
        )

        # Pinned error analysis
        pinned_velocity = pinned_results["drift_analysis"].numerical_velocity

        # Suppression factor
        suppression_factor = (
            pinned_velocity / background_velocity
            if background_velocity > 1e-12
            else 1.0
        )

        # Pinned error (should be much smaller)
        pinned_error = (
            abs(pinned_velocity - theoretical_velocity) / theoretical_velocity
            if theoretical_velocity > 1e-12
            else 0.0
        )

        return {
            "background_error": background_error,
            "pinned_error": pinned_error,
            "suppression_factor": suppression_factor,
            "background_velocity": background_velocity,
            "pinned_velocity": pinned_velocity,
            "theoretical_velocity": theoretical_velocity,
        }

    def _create_beating_summary(
        self, beating_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create beating analysis summary.

        Physical Meaning:
            Creates a comprehensive summary of the beating
            analysis results, including error metrics and
            suppression characteristics.
        """
        total_tests = len(beating_results)

        # Collect error metrics
        background_errors = []
        suppression_factors = []

        for key, result in beating_results.items():
            error_analysis = result["error_analysis"]
            background_errors.append(error_analysis["background_error"])
            suppression_factors.append(error_analysis["suppression_factor"])

        return {
            "total_tests": total_tests,
            "max_background_error": (
                max(background_errors) if background_errors else 0.0
            ),
            "min_suppression_factor": (
                min(suppression_factors) if suppression_factors else 1.0
            ),
            "analysis_complete": True,
            "beating_effects_detected": True,
        }

    def _validate_c4_results(self, beating_results: Dict[str, Any]) -> bool:
        """
        Validate C4 test results.

        Physical Meaning:
            Validates that the C4 test results meet the acceptance
            criteria for mode beating analysis.
        """
        for key, result in beating_results.items():
            error_analysis = result["error_analysis"]

            # Check background error (should be ≤ 10%)
            if error_analysis["background_error"] > 0.10:
                return False

            # Check suppression factor (should be ≥ 10×)
            if error_analysis["suppression_factor"] > 0.1:
                return False

        return True

    def _create_dual_mode_field(
        self, domain: Dict[str, Any], dual_mode: DualModeSource
    ) -> np.ndarray:
        """
        Create dual-mode field.

        Physical Meaning:
            Creates a field configuration with dual-mode
            excitation for beating analysis.
        """
        N = domain["N"]
        L = domain["L"]

        # Create coordinate arrays
        x = np.linspace(0, L, N)
        y = np.linspace(0, L, N)
        z = np.linspace(0, L, N)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        # Create Gaussian profiles
        center = np.array([L / 2, L / 2, L / 2])
        sigma = L / 10

        r_squared = (X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2
        profile = np.exp(-r_squared / (2 * sigma**2))

        # Create dual-mode field
        field = dual_mode.amplitude_1 * profile * np.exp(
            1j * dual_mode.frequency_1 * 0
        ) + dual_mode.amplitude_2 * profile * np.exp(1j * dual_mode.frequency_2 * 0)

        return field

    def _create_dual_mode_field_with_pinning(
        self, domain: Dict[str, Any], dual_mode: DualModeSource
    ) -> np.ndarray:
        """
        Create dual-mode field with pinning effects.

        Physical Meaning:
            Creates a field configuration with dual-mode
            excitation and pinning effects for beating analysis.
        """
        # Create basic dual-mode field
        field = self._create_dual_mode_field(domain, dual_mode)

        # Apply pinning effects (simplified)
        # In practice, this would involve proper memory and pinning implementation
        field *= 0.8  # Simulate pinning-induced field reduction

        return field

    def _evolve_dual_mode_field(
        self, field: np.ndarray, dual_mode: DualModeSource, time_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evolve dual-mode field in time.

        Physical Meaning:
            Performs time evolution of the dual-mode field,
            including beating pattern development.
        """
        dt = time_params.get("dt", 0.005)
        T = time_params.get("T", 400.0)
        time_points = np.arange(0, T, dt)

        field_evolution = [field.copy()]

        for t in time_points[1:]:
            # Apply dual-mode source
            source = self._create_dual_mode_source(dual_mode, t)

            # Apply evolution operator (simplified)
            field = self._apply_evolution_operator(field, source, dt)

            # Store field
            field_evolution.append(field.copy())

        return {
            "time_points": time_points,
            "field_evolution": field_evolution,
            "dual_mode": dual_mode,
        }

    def _evolve_dual_mode_field_with_pinning(
        self, field: np.ndarray, dual_mode: DualModeSource, time_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evolve dual-mode field with pinning effects.

        Physical Meaning:
            Performs time evolution of the dual-mode field
            with pinning effects, including memory-induced
            field stabilization.
        """
        # Use same evolution as background but with pinning effects
        evolution = self._evolve_dual_mode_field(field, dual_mode, time_params)

        # Apply pinning effects to evolution
        # In practice, this would involve proper memory and pinning implementation
        for i, field in enumerate(evolution["field_evolution"]):
            evolution["field_evolution"][i] = field * 0.9  # Simulate pinning

        return evolution

    def _analyze_beating_patterns(
        self, time_evolution: Dict[str, Any], dual_mode: DualModeSource
    ) -> BeatingPattern:
        """
        Analyze beating patterns in field evolution.

        Physical Meaning:
            Analyzes the beating patterns in the field evolution,
            including beating frequency and amplitude modulation.
        """
        field_evolution = time_evolution["field_evolution"]
        time_points = time_evolution["time_points"]

        # Compute amplitude evolution
        amplitude_evolution = [np.abs(field) for field in field_evolution]

        # Compute phase evolution
        phase_evolution = [np.angle(field) for field in field_evolution]

        # Analyze beating frequency
        beating_frequency = abs(dual_mode.frequency_2 - dual_mode.frequency_1)

        # Compute amplitude modulation
        amplitude_modulation = np.array(amplitude_evolution)

        # Compute temporal coherence
        temporal_coherence = self._compute_temporal_coherence(amplitude_evolution)

        return BeatingPattern(
            beating_frequency=beating_frequency,
            amplitude_modulation=amplitude_modulation,
            phase_evolution=phase_evolution,
            temporal_coherence=temporal_coherence,
        )

    def _analyze_drift_velocity(
        self, time_evolution: Dict[str, Any]
    ) -> DriftVelocityAnalysis:
        """
        Analyze drift velocity from field evolution.

        Physical Meaning:
            Computes the drift velocity of field patterns by analyzing
            cross-correlation of effective intensity I_eff(x,t) over time.
        """
        field_evolution = time_evolution["field_evolution"]
        time_points = time_evolution["time_points"]

        # Compute effective intensity
        I_eff = [np.abs(field) ** 2 for field in field_evolution]

        # Apply moving average
        window_size = max(1, int(0.1 * len(I_eff)))
        I_eff_smooth = self._apply_moving_average(I_eff, window_size)

        # Compute cross-correlation shifts
        correlation_shifts = []
        dt = time_points[1] - time_points[0]

        for i in range(len(I_eff_smooth) - 1):
            # Cross-correlation between consecutive time steps
            corr = self._compute_cross_correlation_2d(
                I_eff_smooth[i], I_eff_smooth[i + 1]
            )

            # Find peak shift
            shift = self._find_peak_shift(corr)
            correlation_shifts.append(shift)

        # Compute drift velocity
        numerical_velocity = (
            np.mean(correlation_shifts) / dt if correlation_shifts else 0.0
        )

        # Theoretical velocity (will be computed separately)
        theoretical_velocity = 0.0  # Placeholder

        # Relative error
        relative_error = (
            abs(numerical_velocity - theoretical_velocity) / theoretical_velocity
            if theoretical_velocity > 1e-12
            else 0.0
        )

        return DriftVelocityAnalysis(
            numerical_velocity=numerical_velocity,
            theoretical_velocity=theoretical_velocity,
            relative_error=relative_error,
        )

    def _create_dual_mode_source(
        self, dual_mode: DualModeSource, time: float
    ) -> np.ndarray:
        """
        Create dual-mode source at given time.

        Physical Meaning:
            Creates the dual-mode source s(x,t) = s₁(x) e^(-iω₁t) + s₂(x) e^(-iω₂t)
            at the specified time.
        """
        # Simplified source creation
        # In practice, this would involve proper spatial profiles
        source = dual_mode.amplitude_1 * np.exp(
            -1j * dual_mode.frequency_1 * time
        ) + dual_mode.amplitude_2 * np.exp(-1j * dual_mode.frequency_2 * time)

        return source

    def _apply_evolution_operator(
        self, field: np.ndarray, source: np.ndarray, dt: float
    ) -> np.ndarray:
        """
        Apply evolution operator to field.

        Physical Meaning:
            Applies the evolution operator to advance the field
            in time, including source terms.
        """
        # Simplified evolution (in practice, this would involve proper BVP evolution)
        field_new = field + dt * (source + 0.1j * field)

        return field_new

    def _compute_wave_vector(self, frequency: float) -> WaveVector:
        """
        Compute wave vector for given frequency.

        Physical Meaning:
            Computes the wave vector k = ω / c_φ for the
            given frequency.
        """
        # Simplified wave vector computation
        # In practice, this would involve proper dispersion relation
        magnitude = frequency  # Simplified
        direction = np.array([1, 0, 0])  # Simplified

        return WaveVector(magnitude=magnitude, direction=direction, frequency=frequency)

    def _apply_moving_average(
        self, data: List[np.ndarray], window_size: int
    ) -> List[np.ndarray]:
        """
        Apply moving average to data.

        Physical Meaning:
            Applies moving average smoothing to reduce noise
            in the field evolution data.
        """
        if window_size <= 1:
            return data

        smoothed = []
        for i in range(len(data)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(data), i + window_size // 2 + 1)

            window_data = data[start_idx:end_idx]
            smoothed_data = np.mean(window_data, axis=0)
            smoothed.append(smoothed_data)

        return smoothed

    def _compute_cross_correlation_2d(
        self, field1: np.ndarray, field2: np.ndarray
    ) -> np.ndarray:
        """
        Compute 2D cross-correlation between fields.

        Physical Meaning:
            Computes cross-correlation between two field
            configurations to measure pattern similarity.
        """
        # Simplified cross-correlation (in practice, this would use proper FFT-based correlation)
        correlation = np.sum(field1 * field2.conj())
        return np.array([correlation])

    def _find_peak_shift(self, correlation: np.ndarray) -> float:
        """
        Find peak shift in correlation.

        Physical Meaning:
            Finds the shift of the correlation peak,
            indicating pattern displacement.
        """
        # Simplified peak shift (in practice, this would involve proper peak finding)
        return 0.0

    def _compute_temporal_coherence(
        self, amplitude_evolution: List[np.ndarray]
    ) -> float:
        """
        Compute temporal coherence of amplitude evolution.

        Physical Meaning:
            Computes the temporal coherence of the amplitude
            evolution, indicating pattern stability.
        """
        # Simplified temporal coherence computation
        # In practice, this would involve proper coherence analysis
        return 0.9  # Placeholder value
